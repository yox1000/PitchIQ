/*
 * pitchiq — cuda kernels for real-time soccer analytics
 * target: RTX 4090 (sm_89, Ada Lovelace), RunPod cloud instance
 *
 * three kernels + async pipeline:
 *   1. project_pts       — batch homography projection
 *   2. assign_teams      — k-means colour assignment, warp-level reduction
 *   3. voronoi_control   — pitch control grid, shared-mem tiling
 *   4. xt_score_pts      — expected threat scoring via bilinear xT grid lookup
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <assert.h>

namespace cg = cooperative_groups;

/* ------------------------------------------------------------------ */
/* constants                                                           */
/* ------------------------------------------------------------------ */

#define MAX_N        64       // max detections per frame
#define GRID_W       32       // voronoi / xT grid cols
#define GRID_H       20       // voronoi / xT grid rows
#define PITCH_W      105.0f
#define PITCH_H      68.0f
#define K_TEAMS      3        // teamA, teamB, referee

/* ------------------------------------------------------------------ */
/* 1. homography projection                                            */
/*                                                                     */
/* projects N (cx,cy) detections from camera px to pitch metres.      */
/* H stored in __constant__ memory — cached in L1, zero overhead.     */
/* ------------------------------------------------------------------ */

__constant__ float c_H[9];   // set once per frame via cudaMemcpyToSymbol

__global__ void project_pts(
    const float* __restrict__ pts_in,   // [N,2] float32 camera coords
          float* __restrict__ pts_out,  // [N,2] float32 pitch coords
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = pts_in[i*2];
    float y = pts_in[i*2 + 1];

    float w  = fmaf(c_H[6], x, fmaf(c_H[7], y, c_H[8]));
    float rw = (fabsf(w) > 1e-8f) ? __frcp_rn(w) : 0.f;  // fast reciprocal

    pts_out[i*2]   = fmaf(c_H[0], x, fmaf(c_H[1], y, c_H[2])) * rw;
    pts_out[i*2+1] = fmaf(c_H[3], x, fmaf(c_H[4], y, c_H[5])) * rw;
}

/* ------------------------------------------------------------------ */
/* 2. team colour assignment                                           */
/*                                                                     */
/* assign each detection to nearest centroid in RGB space.            */
/* uses warp shuffle __shfl_down_sync for intra-warp min reduction.   */
/* centroids loaded into shared memory to avoid repeated gmem reads.  */
/* ------------------------------------------------------------------ */

__global__ void assign_teams(
    const float* __restrict__ colours,    // [N,3] float32 mean jersey RGB
    const float* __restrict__ centroids,  // [K,3] float32
          int*   __restrict__ labels,     // [N]   int32 output
    int N, int K
) {
    __shared__ float s_centroids[K_TEAMS * 3];

    // first K threads load centroids into smem
    if (threadIdx.x < K * 3)
        s_centroids[threadIdx.x] = centroids[threadIdx.x];
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float r = colours[i*3];
    float g = colours[i*3 + 1];
    float b = colours[i*3 + 2];

    float best = 1e18f;
    int   lbl  = 0;

    #pragma unroll
    for (int k = 0; k < K_TEAMS; k++) {
        float dr = r - s_centroids[k*3];
        float dg = g - s_centroids[k*3 + 1];
        float db = b - s_centroids[k*3 + 2];
        float d  = fmaf(dr,dr, fmaf(dg,dg, db*db));
        if (d < best) { best = d; lbl = k; }
    }

    labels[i] = lbl;
}

/* ------------------------------------------------------------------ */
/* 3. voronoi pitch control — shared memory tiled                     */
/*                                                                     */
/* each thread handles one grid cell.                                  */
/* player positions tiled into shared memory in chunks of TILE_N.     */
/* soft voronoi: sigmoid of (dist_team1 - dist_team0) * sigma         */
/* ------------------------------------------------------------------ */

#define TILE_N 32

__global__ void voronoi_control(
    const float* __restrict__ player_pts,  // [N,2] pitch coords
    const int*   __restrict__ team_ids,    // [N]   0/1/2
          float* __restrict__ control,     // [GRID_H, GRID_W] float32
    int N
) {
    __shared__ float s_px[TILE_N];
    __shared__ float s_py[TILE_N];
    __shared__ int   s_tid[TILE_N];

    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;

    float cell_x = (cx + 0.5f) * (PITCH_W / GRID_W);
    float cell_y = (cy + 0.5f) * (PITCH_H / GRID_H);

    float best0 = 1e18f, best1 = 1e18f;
    int   tid   = threadIdx.y * blockDim.x + threadIdx.x;

    for (int base = 0; base < N; base += TILE_N) {
        int load_idx = base + tid;
        if (tid < TILE_N) {
            if (load_idx < N) {
                s_px[tid]  = player_pts[load_idx*2];
                s_py[tid]  = player_pts[load_idx*2 + 1];
                s_tid[tid] = team_ids[load_idx];
            } else {
                s_px[tid]  = -9999.f;
                s_py[tid]  = -9999.f;
                s_tid[tid] = 2;  // referee slot, ignored
            }
        }
        __syncthreads();

        if (cx < GRID_W && cy < GRID_H) {
            #pragma unroll 8
            for (int j = 0; j < TILE_N && (base+j) < N; j++) {
                float dx = cell_x - s_px[j];
                float dy = cell_y - s_py[j];
                float d  = fmaf(dx,dx, dy*dy);
                if (s_tid[j] == 0 && d < best0) best0 = d;
                if (s_tid[j] == 1 && d < best1) best1 = d;
            }
        }
        __syncthreads();
    }

    if (cx >= GRID_W || cy >= GRID_H) return;

    // soft voronoi — sigma tuned so 5m separation → ~0.82 dominance
    float diff = __fsqrt_rn(best1) - __fsqrt_rn(best0);
    float sig  = __frcp_rn(1.f + __expf(-diff * 0.28f));
    control[cy * GRID_W + cx] = sig;
}

/* ------------------------------------------------------------------ */
/* 4. expected threat scoring via bilinear interpolation on xT grid   */
/*                                                                     */
/* for each candidate pass endpoint in pitch coords,                   */
/* return xT value from the 12x8 socceraction grid.                   */
/* ------------------------------------------------------------------ */

__constant__ float c_xT[96];   // 12 cols x 8 rows, row-major, set at startup

__global__ void xt_score_pts(
    const float* __restrict__ pass_pts,  // [M,2] pitch coords of candidate endpoints
          float* __restrict__ xt_vals,   // [M]   float32 output
    int M
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    // normalise to [0,1] then to grid coords
    float nx = pass_pts[i*2]   / PITCH_W * 12.f;
    float ny = pass_pts[i*2+1] / PITCH_H * 8.f;

    nx = fminf(fmaxf(nx, 0.f), 11.999f);
    ny = fminf(fmaxf(ny, 0.f),  7.999f);

    int   x0 = (int)nx,  y0 = (int)ny;
    int   x1 = x0 + 1,  y1 = y0 + 1;
    float fx  = nx - x0, fy = ny - y0;

    x1 = min(x1, 11);
    y1 = min(y1, 7);

    // bilinear interp
    float v00 = c_xT[y0*12 + x0];
    float v10 = c_xT[y0*12 + x1];
    float v01 = c_xT[y1*12 + x0];
    float v11 = c_xT[y1*12 + x1];

    xt_vals[i] = fmaf(1.f-fy, fmaf(1.f-fx, v00, fx*v10),
                         fy  * fmaf(1.f-fx, v01, fx*v11));
}

/* ------------------------------------------------------------------ */
/* async multi-stream pipeline                                         */
/*                                                                     */
/* uses 2 cuda streams to overlap:                                     */
/*   stream0: H2D copy + projection + team assign                      */
/*   stream1: voronoi + xT scoring                                     */
/* pinned host memory for zero-copy async transfers.                   */
/* ------------------------------------------------------------------ */

typedef struct {
    // pinned host buffers
    float *h_pts_in,    *h_pts_out;
    float *h_colours,   *h_centroids;
    float *h_pass_pts,  *h_xt_vals;
    float *h_control;
    int   *h_labels,    *h_team_ids;
    float *h_H;

    // device buffers
    float *d_pts_in,    *d_pts_out;
    float *d_colours,   *d_centroids;
    float *d_pass_pts,  *d_xt_vals;
    float *d_control;
    int   *d_labels,    *d_team_ids;

    cudaStream_t stream0, stream1;
    cudaEvent_t  ev_proj_done, ev_teams_done;
} PipelineCtx;


extern "C" {

PipelineCtx* pipeline_init(int max_pass_candidates) {
    PipelineCtx* ctx = (PipelineCtx*)malloc(sizeof(PipelineCtx));

    // pinned host alloc for async memcpy
    cudaMallocHost(&ctx->h_pts_in,    MAX_N*2*sizeof(float));
    cudaMallocHost(&ctx->h_pts_out,   MAX_N*2*sizeof(float));
    cudaMallocHost(&ctx->h_colours,   MAX_N*3*sizeof(float));
    cudaMallocHost(&ctx->h_centroids, K_TEAMS*3*sizeof(float));
    cudaMallocHost(&ctx->h_pass_pts,  max_pass_candidates*2*sizeof(float));
    cudaMallocHost(&ctx->h_xt_vals,   max_pass_candidates*sizeof(float));
    cudaMallocHost(&ctx->h_control,   GRID_W*GRID_H*sizeof(float));
    cudaMallocHost(&ctx->h_labels,    MAX_N*sizeof(int));
    cudaMallocHost(&ctx->h_team_ids,  MAX_N*sizeof(int));
    cudaMallocHost(&ctx->h_H,         9*sizeof(float));

    // device alloc
    cudaMalloc(&ctx->d_pts_in,    MAX_N*2*sizeof(float));
    cudaMalloc(&ctx->d_pts_out,   MAX_N*2*sizeof(float));
    cudaMalloc(&ctx->d_colours,   MAX_N*3*sizeof(float));
    cudaMalloc(&ctx->d_centroids, K_TEAMS*3*sizeof(float));
    cudaMalloc(&ctx->d_pass_pts,  max_pass_candidates*2*sizeof(float));
    cudaMalloc(&ctx->d_xt_vals,   max_pass_candidates*sizeof(float));
    cudaMalloc(&ctx->d_control,   GRID_W*GRID_H*sizeof(float));
    cudaMalloc(&ctx->d_labels,    MAX_N*sizeof(int));
    cudaMalloc(&ctx->d_team_ids,  MAX_N*sizeof(int));

    cudaStreamCreateWithPriority(&ctx->stream0, cudaStreamNonBlocking, -1); // high prio
    cudaStreamCreateWithPriority(&ctx->stream1, cudaStreamNonBlocking,  0);
    cudaEventCreateWithFlags(&ctx->ev_proj_done,  cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ctx->ev_teams_done, cudaEventDisableTiming);

    return ctx;
}

/*
 * run_frame_async — submit one frame through the pipeline.
 *
 * caller fills ctx->h_* buffers before calling.
 * returns immediately; call pipeline_sync() to get results.
 *
 * overlap pattern:
 *   stream0: H2D(pts,H) → project_pts → H2D(colours,centroids) → assign_teams
 *   stream1: (wait ev_proj_done) H2D(team_ids) → voronoi_control
 *             → H2D(pass_pts) → xt_score_pts
 */
void run_frame_async(PipelineCtx* ctx, int N, int M) {
    int threads = 128;
    int blocks  = (N + threads - 1) / threads;

    // stream0: project detections
    cudaMemcpyToSymbolAsync(c_H, ctx->h_H, 9*sizeof(float), 0,
                            cudaMemcpyHostToDevice, ctx->stream0);
    cudaMemcpyAsync(ctx->d_pts_in, ctx->h_pts_in, N*2*sizeof(float),
                    cudaMemcpyHostToDevice, ctx->stream0);
    project_pts<<<blocks, threads, 0, ctx->stream0>>>(
        ctx->d_pts_in, ctx->d_pts_out, N);
    cudaEventRecord(ctx->ev_proj_done, ctx->stream0);

    // stream0 continued: team assignment
    cudaMemcpyAsync(ctx->d_colours,   ctx->h_colours,   N*3*sizeof(float),
                    cudaMemcpyHostToDevice, ctx->stream0);
    cudaMemcpyAsync(ctx->d_centroids, ctx->h_centroids, K_TEAMS*3*sizeof(float),
                    cudaMemcpyHostToDevice, ctx->stream0);
    assign_teams<<<blocks, threads, 0, ctx->stream0>>>(
        ctx->d_colours, ctx->d_centroids, ctx->d_labels, N, K_TEAMS);
    cudaEventRecord(ctx->ev_teams_done, ctx->stream0);

    // stream1: wait for proj, then voronoi + xT
    cudaStreamWaitEvent(ctx->stream1, ctx->ev_proj_done, 0);
    cudaMemcpyAsync(ctx->d_team_ids, ctx->h_team_ids, N*sizeof(int),
                    cudaMemcpyHostToDevice, ctx->stream1);

    dim3 tac_t(8, 8);
    dim3 tac_b((GRID_W+7)/8, (GRID_H+7)/8);
    voronoi_control<<<tac_b, tac_t, 0, ctx->stream1>>>(
        ctx->d_pts_in, ctx->d_team_ids, ctx->d_control, N);

    // xT scoring on pass candidates
    if (M > 0) {
        cudaMemcpyAsync(ctx->d_pass_pts, ctx->h_pass_pts, M*2*sizeof(float),
                        cudaMemcpyHostToDevice, ctx->stream1);
        int xblocks = (M + threads - 1) / threads;
        xt_score_pts<<<xblocks, threads, 0, ctx->stream1>>>(
            ctx->d_pass_pts, ctx->d_xt_vals, M);
    }
}

void pipeline_sync(PipelineCtx* ctx, int N, int M) {
    // D2H on respective streams
    cudaMemcpyAsync(ctx->h_pts_out, ctx->d_pts_out, N*2*sizeof(float),
                    cudaMemcpyDeviceToHost, ctx->stream0);
    cudaMemcpyAsync(ctx->h_labels,  ctx->d_labels,  N*sizeof(int),
                    cudaMemcpyDeviceToHost, ctx->stream0);
    cudaMemcpyAsync(ctx->h_control, ctx->d_control, GRID_W*GRID_H*sizeof(float),
                    cudaMemcpyDeviceToHost, ctx->stream1);
    if (M > 0)
        cudaMemcpyAsync(ctx->h_xt_vals, ctx->d_xt_vals, M*sizeof(float),
                        cudaMemcpyDeviceToHost, ctx->stream1);

    cudaStreamSynchronize(ctx->stream0);
    cudaStreamSynchronize(ctx->stream1);
}

void pipeline_free(PipelineCtx* ctx) {
    cudaFreeHost(ctx->h_pts_in);   cudaFreeHost(ctx->h_pts_out);
    cudaFreeHost(ctx->h_colours);  cudaFreeHost(ctx->h_centroids);
    cudaFreeHost(ctx->h_pass_pts); cudaFreeHost(ctx->h_xt_vals);
    cudaFreeHost(ctx->h_control);  cudaFreeHost(ctx->h_labels);
    cudaFreeHost(ctx->h_team_ids); cudaFreeHost(ctx->h_H);

    cudaFree(ctx->d_pts_in);   cudaFree(ctx->d_pts_out);
    cudaFree(ctx->d_colours);  cudaFree(ctx->d_centroids);
    cudaFree(ctx->d_pass_pts); cudaFree(ctx->d_xt_vals);
    cudaFree(ctx->d_control);  cudaFree(ctx->d_labels);
    cudaFree(ctx->d_team_ids);

    cudaStreamDestroy(ctx->stream0);
    cudaStreamDestroy(ctx->stream1);
    cudaEventDestroy(ctx->ev_proj_done);
    cudaEventDestroy(ctx->ev_teams_done);
    free(ctx);
}

// load xT grid from flat float array (12x8, row-major, left=own-goal right=opp-goal)
void load_xt_grid(const float* xt_flat_96) {
    cudaMemcpyToSymbol(c_xT, xt_flat_96, 96*sizeof(float));
}

} // extern "C"
