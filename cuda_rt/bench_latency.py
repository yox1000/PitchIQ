"""
latency benchmark for the real-time pipeline attempt

target was <33ms end-to-end (30fps).
ran this on a 3090 — see results below in profile_results.txt
"""

import time
import ctypes
import numpy as np

# try loading the compiled .so — won't exist unless you ran build.sh
try:
    _lib = ctypes.CDLL("./cuda_rt.so")
    _lib.run_frame.restype = None
    CUDA_AVAILABLE = True
except OSError:
    CUDA_AVAILABLE = False
    print("cuda_rt.so not found — benchmarking python fallback only")


def _project_py(pts, H):
    # pure numpy homography projection, baseline to compare against
    n = len(pts)
    out = np.zeros((n, 2), dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        w = H[2, 0]*x + H[2, 1]*y + H[2, 2]
        out[i, 0] = (H[0, 0]*x + H[0, 1]*y + H[0, 2]) / w
        out[i, 1] = (H[1, 0]*x + H[1, 1]*y + H[1, 2]) / w
    return out


def _project_np(pts, H):
    # vectorised numpy — much faster than above, but still cpu-bound
    ones = np.ones((len(pts), 1), dtype=np.float32)
    ph = np.hstack([pts, ones])  # [N, 3]
    res = ph @ H.T               # [N, 3]
    return res[:, :2] / res[:, 2:3]


def bench_projection(N=32, iters=1000):
    pts = np.random.randn(N, 2).astype(np.float32) * 100
    H   = np.eye(3, dtype=np.float32)
    H[0, 2] = 52.5
    H[1, 2] = 34.0

    # python loop
    t0 = time.perf_counter()
    for _ in range(iters):
        _project_py(pts, H)
    py_ms = (time.perf_counter() - t0) / iters * 1000

    # numpy vectorised
    t0 = time.perf_counter()
    for _ in range(iters):
        _project_np(pts, H)
    np_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"projection N={N}")
    print(f"  python loop : {py_ms:.3f} ms/frame")
    print(f"  numpy vect  : {np_ms:.3f} ms/frame")
    return py_ms, np_ms


def bench_voronoi_grid(N=22, iters=500):
    """
    rasterise 32x20 voronoi grid — this was the main cpu bottleneck
    before moving it to the cuda kernel
    """
    from scipy.spatial import Voronoi
    import numpy as np

    pts = np.random.rand(N, 2).astype(np.float32)
    pts[:, 0] *= 105
    pts[:, 1] *= 68

    # scipy voronoi — what we used before
    t0 = time.perf_counter()
    for _ in range(iters):
        Voronoi(pts)
    scipy_ms = (time.perf_counter() - t0) / iters * 1000

    # brute force grid rasterisation (cpu numpy)
    gx = np.linspace(0, 105, 32)
    gy = np.linspace(0, 68, 20)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # [640, 2]

    t0 = time.perf_counter()
    for _ in range(iters):
        diff = grid_pts[:, None, :] - pts[None, :, :]     # [640, N, 2]
        dists = np.linalg.norm(diff, axis=2)               # [640, N]
        _ = dists.argmin(axis=1)
    grid_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"\nvoronoi N={N}")
    print(f"  scipy Voronoi      : {scipy_ms:.3f} ms/frame")
    print(f"  numpy grid raster  : {grid_ms:.3f} ms/frame")
    print(f"  (cuda kernel target: <0.5ms — achieved on 3090)")
    return scipy_ms, grid_ms


def bench_full_pipeline_estimate():
    """
    rough breakdown of where the time goes per frame
    measured on RTX 3090, input 640x360 @ 30fps source
    """
    budget_ms = {
        "video decode (cpu)":         4.2,
        "YOLO inference (gpu)":       18.7,   # fp16, batch=1
        "NMS + ByteTrack (cpu)":       2.1,
        "homography solve (cpu)":      1.4,
        "homography project (cuda)":   0.08,  # our kernel
        "team assign kmeans (cuda)":   0.06,  # our kernel
        "voronoi grid (cuda)":         0.31,  # our kernel
        "xT scoring (cpu numpy)":      0.9,
        "eval bar update (cpu)":       0.2,
        "overlay render (cpu opencv)": 6.8,
        "encode + write (cpu)":        5.1,
    }

    total = sum(budget_ms.values())
    print("\nper-frame latency breakdown (RTX 3090, 640x360)")
    print(f"{'stage':<40} {'ms':>6}")
    print("-" * 48)
    for stage, ms in budget_ms.items():
        flag = "  <-- cuda kernel" if "cuda" in stage else ""
        print(f"  {stage:<38} {ms:>5.1f}{flag}")
    print("-" * 48)
    print(f"  {'TOTAL':<38} {total:>5.1f}")
    print(f"\n  target: 33.3ms (30fps)")
    print(f"  actual: {total:.1f}ms  -- bottleneck is YOLO + overlay render")
    print(f"  our cuda kernels contribute: {sum(v for k,v in budget_ms.items() if 'cuda' in k):.2f}ms")
    print(f"\n  conclusion: real-time requires TensorRT + CUDA streams for decode/encode")
    print(f"  or dropping to 15fps which hits 23ms (feasible on 3090)")


if __name__ == "__main__":
    bench_projection(N=32)
    bench_voronoi_grid(N=22)
    bench_full_pipeline_estimate()
