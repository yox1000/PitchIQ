/*
 * batch homography projection — CUDA kernel
 *
 * goal: project N player detections from camera-space to 2D pitch coords
 * in a single kernel launch instead of looping in python per-frame.
 *
 * turns out the bottleneck isn't the projection math, it's the D2H copy
 * and YOLO inference itself. see notes.md
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_DETECTIONS 64

/*
 * each thread handles one detection.
 * H is the 3x3 homography matrix, flat row-major.
 * pts_in:  [N, 2] float32  (cx, cy in pixel space)
 * pts_out: [N, 2] float32  (x, y in pitch metres)
 */
__global__ void project_pts(
    const float* __restrict__ pts_in,
    float*       __restrict__ pts_out,
    const float* __restrict__ H,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = pts_in[i * 2];
    float y = pts_in[i * 2 + 1];

    float w = H[6]*x + H[7]*y + H[8];
    // guard against degenerate homography
    if (fabsf(w) < 1e-8f) {
        pts_out[i * 2]     = 0.f;
        pts_out[i * 2 + 1] = 0.f;
        return;
    }

    pts_out[i * 2]     = (H[0]*x + H[1]*y + H[2]) / w;
    pts_out[i * 2 + 1] = (H[3]*x + H[4]*y + H[5]) / w;
}

/*
 * team colour assignment kernel — k-means step
 *
 * assign each detection to nearest centroid by L2 in RGB space.
 * centroids: [K, 3]  float32
 * colours:   [N, 3]  float32  (mean jersey colour from bounding box crop)
 * labels:    [N]     int32    output
 */
__global__ void assign_teams(
    const float* __restrict__ colours,
    const float* __restrict__ centroids,
    int*         __restrict__ labels,
    int N,
    int K
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float r = colours[i*3];
    float g = colours[i*3 + 1];
    float b = colours[i*3 + 2];

    float best_dist = 1e18f;
    int   best_k    = 0;

    for (int k = 0; k < K; k++) {
        float dr = r - centroids[k*3];
        float dg = g - centroids[k*3 + 1];
        float db = b - centroids[k*3 + 2];
        float d  = dr*dr + dg*dg + db*db;
        if (d < best_dist) {
            best_dist = d;
            best_k    = k;
        }
    }

    labels[i] = best_k;
}

/*
 * voronoi dominance — rasterise pitch control grid
 *
 * for each grid cell, find nearest player per team, compute dominance.
 * grid: GRID_H x GRID_W cells covering 105x68m pitch
 *
 * player_pts: [N, 2] float32  pitch coords
 * team_ids:   [N]    int32    0 or 1
 * control:    [GRID_H, GRID_W] float32   output  0=team0, 1=team1, 0.5=equal
 */
#define GRID_W 32
#define GRID_H 20
#define PITCH_W 105.f
#define PITCH_H 68.f

__global__ void voronoi_control(
    const float* __restrict__ player_pts,
    const int*   __restrict__ team_ids,
    float*       __restrict__ control,
    int N
) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= GRID_W || cy >= GRID_H) return;

    float px = (cx + 0.5f) * (PITCH_W / GRID_W);
    float py = (cy + 0.5f) * (PITCH_H / GRID_H);

    float best0 = 1e18f, best1 = 1e18f;

    for (int i = 0; i < N; i++) {
        float dx = px - player_pts[i*2];
        float dy = py - player_pts[i*2 + 1];
        float d  = dx*dx + dy*dy;
        if (team_ids[i] == 0 && d < best0) best0 = d;
        if (team_ids[i] == 1 && d < best1) best1 = d;
    }

    // soft voronoi: sigmoid of distance difference
    float diff = sqrtf(best1) - sqrtf(best0);
    control[cy * GRID_W + cx] = 1.f / (1.f + expf(-diff * 0.3f));
}


/* ---------- host wrapper used from python via ctypes ---------- */

extern "C" {

void run_frame(
    float* h_pts_in,
    float* h_pts_out,
    float* h_H,
    float* h_colours,
    int*   h_labels,
    float* h_centroids,
    float* h_control,
    int*   h_team_ids,
    int    N,
    int    K
) {
    float *d_pts_in, *d_pts_out, *d_H, *d_colours, *d_centroids, *d_control;
    int   *d_labels, *d_team_ids;

    cudaMalloc(&d_pts_in,    N*2*sizeof(float));
    cudaMalloc(&d_pts_out,   N*2*sizeof(float));
    cudaMalloc(&d_H,         9*sizeof(float));
    cudaMalloc(&d_colours,   N*3*sizeof(float));
    cudaMalloc(&d_centroids, K*3*sizeof(float));
    cudaMalloc(&d_labels,    N*sizeof(int));
    cudaMalloc(&d_team_ids,  N*sizeof(int));
    cudaMalloc(&d_control,   GRID_W*GRID_H*sizeof(float));

    cudaMemcpy(d_pts_in,    h_pts_in,    N*2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_H,         h_H,         9*sizeof(float),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_colours,   h_colours,   N*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_team_ids,  h_team_ids,  N*sizeof(int),     cudaMemcpyHostToDevice);

    int threads = 64;
    int blocks  = (N + threads - 1) / threads;

    project_pts<<<blocks, threads>>>(d_pts_in, d_pts_out, d_H, N);
    assign_teams<<<blocks, threads>>>(d_colours, d_centroids, d_labels, N, K);

    dim3 tac_threads(8, 8);
    dim3 tac_blocks((GRID_W+7)/8, (GRID_H+7)/8);
    voronoi_control<<<tac_blocks, tac_threads>>>(d_pts_in, d_team_ids, d_control, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_pts_out, d_pts_out, N*2*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_labels,  d_labels,  N*sizeof(int),     cudaMemcpyDeviceToHost);
    cudaMemcpy(h_control, d_control, GRID_W*GRID_H*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_pts_in); cudaFree(d_pts_out); cudaFree(d_H);
    cudaFree(d_colours); cudaFree(d_centroids); cudaFree(d_labels);
    cudaFree(d_team_ids); cudaFree(d_control);
}

}
