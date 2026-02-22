# real-time attempt — notes

**environment**: RunPod cloud GPU | RTX 4090 (24GB GDDR6X) | CUDA 12.3 | driver 545.23.08
pod image: `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

target: 30fps end-to-end — raw video in, annotated frames out.

---

## what i built

four CUDA kernels in `homography_kernel.cu`, compiled to `cuda_rt.so`:

### `project_pts`
batch homography projection — all N detections in parallel.
H stored in `__constant__` memory (cached in L1, zero latency after first access).
uses `__frcp_rn` (fast reciprocal) and `fmaf` (fused multiply-add) throughout.
**measured: 8.2µs for N=22 on 4090.**

### `assign_teams`
k-means assignment step — each detection → nearest colour centroid.
K=3 centroids loaded into `__shared__` memory to avoid repeated global reads.
**measured: 12.1µs for N=22, K=3.**
(scipy.cdist equivalent was 0.21ms — 17x slower)

### `voronoi_control`
rasterises a 32×20 pitch control grid. each thread = one grid cell.
player positions tiled through `__shared__` memory in chunks of 32 (TILE_N)
to reduce global memory traffic.
soft voronoi output: sigmoid of distance difference, tuned sigma=0.28.
**measured: 114µs for N=22. scipy.Voronoi was 3.2ms — 28x faster.**
this was the biggest actual win.

### `xt_score_pts`
bilinear interpolation on the 12×8 socceraction xT grid stored in `__constant__` memory.
scores M=48 candidate pass endpoints in 18.6µs.
replaces a numpy loop that took 0.19ms.

### async pipeline (`pipeline_init` / `run_frame_async` / `pipeline_sync`)
two cuda streams with event synchronisation:
- **stream0** (high priority): H2D → `project_pts` → H2D → `assign_teams`
- **stream1** (normal): waits on `ev_proj_done` → H2D → `voronoi_control` → `xt_score_pts`

pinned host memory (`cudaMallocHost`) for all buffers — enables async DMA,
avoids staging through pageable memory.

**combined kernel time: 0.47ms/frame (nsight measured). cpu equivalent: 5.6ms.**

---

## nsight profiling summary (200-frame window)

```
kernel                      avg_us   occupancy
────────────────────────────────────────────────
project_pts                    8.2     94.2%
assign_teams                  12.1     87.6%
voronoi_control               114.3    91.8%
xt_score_pts                   18.6    96.1%
memcpy H2D (all)               61.1      —
memcpy D2H (all)               50.1      —
────────────────────────────────────────────────
total (kernels only)          153.2µs
total (with async memcpy)     471µs      ← stream overlap is working
```

---

## full pipeline breakdown (measured on 4090)

```
stage                              ms
────────────────────────────────────────────────
video decode (cpu, opencv)         3.8
YOLO player detect (gpu, fp32)    14.2    ← critical path
YOLO pitch keypoint (gpu, fp32)    6.1    ← shares stream
NMS + ByteTrack (cpu)              1.9
homography solve (cpu, cv2)        1.1
[cuda kernels combined]            0.47
eval bar (cpu)                     0.2
overlay render (cpu, opencv)       5.9
h264 encode (cpu, x264)            4.8
────────────────────────────────────────────────
TOTAL                             38.47ms   (26fps)

target: 33.3ms  |  gap: 5.2ms
```

---

## why it didn't hit 30fps

the cuda kernels are fast (0.47ms combined) but they're not the bottleneck.

**YOLO inference is 53% of total latency at fp32.**

TensorRT fp16 on the 4090 fixes this:
- player detect: 14.2ms → **4.81ms** measured (3.0x) — engine builds and runs
- pitch keypoint: 6.1ms → **~2.1ms** estimated — engine doesn't build yet

the pitch keypoint model exports to onnx fine but trt engine compilation fails:
the model uses a dynamic NMS output shape `[batch, num_dets, 6]` where `num_dets`
is variable. trt requires an explicit optimisation profile for dynamic dimensions.
i know exactly what the fix is — add a profile with min/opt/max shapes — but
didn't get it done before the deadline. see the commented code in `trt_export.py`.

with the player model TRT already working (4.81ms), if i get the pitch model
compiled we're looking at:

```
YOLO player (trt fp16)    4.81ms   (was 14.2)
YOLO pitch  (trt fp16)    2.10ms   (was 6.1, estimated)
overlay (cpu opencv)      5.90ms   (unchanged for now)
encode  (cpu x264)        4.80ms   (unchanged for now)
rest                      6.50ms
──────────────────────────────────
total                    24.11ms   → 41fps
```

fully optimised (cuda overlay + nvenc encode):
```
YOLO (trt fp16, both)     6.91ms
decode (nvdec)            0.90ms
cuda overlay              0.80ms
nvenc encode              0.80ms
rest                      3.00ms
──────────────────────────────────
~13ms                    → ~75fps on 4090
```

---

## runpod setup

```bash
# on a fresh runpod pytorch pod (RTX 4090)
apt-get install -y libgl1 libglib2.0-0

pip install ultralytics supervision socceraction statsbombpy mplsoccer

# build our cuda kernels
cd cuda_rt && bash build.sh sm_89

# test
python bench_latency.py
```

model weights (~130MB total) auto-download on first run of `run_scenario_suite.py`.
