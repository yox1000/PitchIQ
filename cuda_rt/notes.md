# real-time attempt notes

target: 30fps end-to-end on a single RTX 3090

## what i built

three cuda kernels in `homography_kernel.cu`:

1. **`project_pts`** — batch homography projection, N detections in parallel.
   trivial math but eliminates a python loop that ran per-frame. now 0.08ms.

2. **`assign_teams`** — k-means assignment step (colour → team label) on gpu.
   K=3 (teamA, teamB, referee), N≤64 detections. runs in 0.06ms.

3. **`voronoi_control`** — rasterise 32×20 pitch control grid.
   brute force nearest-player per cell, parallelised over grid cells.
   was 3.8ms with scipy, now 0.31ms. biggest actual win.

compiled via `build.sh`, loaded from python via ctypes.
see `bench_latency.py` for measured results.

## why it still didn't hit 30fps

total per-frame: ~39.9ms (target: 33.3ms)

the cuda kernels are fast (0.5ms combined) but the pipeline has bigger
bottlenecks that aren't in our code:

- **yolo inference: 18.7ms** — fp32, ultralytics native. this is the killer.
  fp16 TensorRT would cut this to ~6ms. started `trt_export.py` but the
  pitch keypoint model has a dynamic output shape that trt refuses to
  compile without explicit optimization profiles. didn't finish.

- **overlay render: 6.8ms** — cv2 drawing on cpu. could move to cuda but
  the drawing primitives are annoying to implement (circles, lines, text).

- **encode: 5.1ms** — software x264. nvenc would be ~1ms.

## path to 30fps (unfinished)

1. export yolo models to trt fp16 → -12ms
2. cuda overlay render → -5ms
3. nvenc encode → -4ms
4. cuda streams to overlap stages → another -3ms

rough estimate: gets to ~16ms (60fps headroom).
would need a week to implement properly.

## what actually runs today

the kernels compile and run. `bench_latency.py` shows the voronoi speedup.
the trt export doesn't complete (pitch model shape issue).

the offline pipeline (run_scenario_suite.py) does not use these kernels —
it still uses scipy/numpy. the kernels are the realtime branch that didn't merge.
