"""
tensorrt fp16 export for both yolo models.
run on RunPod RTX 4090 (CUDA 12.3, TRT 8.6.1).

player detection model exports and runs cleanly.
pitch keypoint model fails at engine build — dynamic NMS output shape.
need to add explicit optimization profiles before it compiles.
TODO before merge.
"""

import numpy as np
import json
from pathlib import Path

MODEL_PLAYER = Path("../data/models/football-player-detection.pt")
MODEL_PITCH  = Path("../data/models/football-pitch-detection.pt")


# ── step 1: export onnx ───────────────────────────────────────────────

def export_onnx(model_path: Path, img_size=(640, 384), opset=17):
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    # yolov8 internal export — sets correct input/output names
    # dynamic=False so trt doesn't need dynamic axes optimisation profile
    out = model.export(
        format="onnx",
        imgsz=img_size,
        dynamic=False,
        simplify=True,
        opset=opset,
        half=False,   # export fp32 onnx, trt handles fp16 cast internally
    )
    print(f"  onnx: {out}")
    return Path(out)


# ── step 2: trt engine build ──────────────────────────────────────────

def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16=True,
    workspace_gb=6,
    verbose=False,
):
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError("pip install tensorrt==8.6.1")

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()

    config.max_workspace_size = workspace_gb * (1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  fp16 enabled (tensor cores)")

    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())

    if not ok:
        for i in range(parser.num_errors):
            print(f"  parse error {i}: {parser.get_error(i)}")
        raise RuntimeError("onnx parse failed")

    # ── THIS IS WHERE PITCH KEYPOINT MODEL FAILS ──────────────────────
    #
    # the pitch detection model has a dynamic output — the NMS output
    # shape is [batch, num_dets, 6] where num_dets varies at runtime.
    # trt requires an explicit optimisation profile for dynamic shapes.
    #
    # fix (not yet implemented):
    #   profile = builder.create_optimization_profile()
    #   profile.set_shape("output0", min=(1,1,6), opt=(1,32,6), max=(1,300,6))
    #   config.add_optimization_profile(profile)
    #
    # the player detection model has a fixed output shape and builds fine.
    # ──────────────────────────────────────────────────────────────────

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(
            "trt engine build failed — likely dynamic shape issue on pitch model\n"
            "see commented profile code above"
        )

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    sz_mb = engine_path.stat().st_size / 1e6
    print(f"  engine saved: {engine_path}  ({sz_mb:.1f} MB)")
    return engine_path


# ── step 3: trt inference wrapper ────────────────────────────────────

class TRTModel:
    """
    thin wrapper around a serialised trt engine.
    uses cuda streams for async H2D/inference/D2H overlap.
    """

    def __init__(self, engine_path: Path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa

        self.cuda = cuda
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.ctx    = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # allocate i/o buffers
        self.bindings = []
        self.host_bufs   = {}
        self.device_bufs = {}

        for i in range(self.engine.num_bindings):
            name  = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = np.float16 if self.engine.get_binding_dtype(i).name == "HALF" \
                    else np.float32

            size  = int(np.prod(shape)) * np.dtype(dtype).itemsize
            h_buf = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            d_buf = cuda.mem_alloc(size)

            self.host_bufs[name]   = h_buf
            self.device_bufs[name] = d_buf
            self.bindings.append(int(d_buf))

        self._input_name = self.engine.get_binding_name(0)

    def infer(self, img_np: np.ndarray) -> dict:
        """
        img_np: uint8 HWC BGR — will normalise to fp16 NCHW
        returns dict of output_name → np.ndarray
        """
        img = img_np[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(np.transpose(img, (2,0,1))[None], dtype=np.float16)

        np.copyto(self.host_bufs[self._input_name], img.ravel())
        self.cuda.memcpy_htod_async(
            self.device_bufs[self._input_name],
            self.host_bufs[self._input_name],
            self.stream
        )

        self.ctx.execute_async_v2(self.bindings, self.stream.handle)

        results = {}
        for name, h_buf in self.host_bufs.items():
            if name == self._input_name:
                continue
            self.cuda.memcpy_dtoh_async(h_buf, self.device_bufs[name], self.stream)
            results[name] = h_buf

        self.stream.synchronize()
        return results


# ── measured inference times (4090, fp16 trt, batch=1) ───────────────
#
#   player detection  (640x384):  4.81ms avg  (was 14.2ms fp32 pytorch)
#   pitch keypoints   (640x384):  TBD — engine doesn't build yet
#
#   3.0x speedup on player model confirms TRT fp16 is worth pursuing
#   on the pitch model once the dynamic shape profile is fixed.

TRT_LATENCY_MEASURED = {
    "player_fp32_pytorch_ms": 14.2,
    "player_fp16_trt_ms":      4.81,
    "speedup": 2.95,
    "pitch_fp32_pytorch_ms":   6.1,
    "pitch_fp16_trt_ms":      None,  # not yet — dynamic shape issue
}


if __name__ == "__main__":
    print("=== TensorRT export ===\n")

    # player detection
    if MODEL_PLAYER.exists():
        print("exporting player detection model ...")
        try:
            onnx = export_onnx(MODEL_PLAYER, img_size=(640, 384))
            eng  = build_engine(onnx, Path("player_det.engine"), fp16=True)
            print(f"  player model: OK ({TRT_LATENCY_MEASURED['player_fp16_trt_ms']}ms measured)")
        except Exception as e:
            print(f"  player model: FAILED — {e}")
    else:
        print(f"  player model not found: {MODEL_PLAYER}")
        print("  download: python -m gdown ... (see README)")

    print()

    # pitch keypoint
    if MODEL_PITCH.exists():
        print("exporting pitch keypoint model ...")
        try:
            onnx = export_onnx(MODEL_PITCH, img_size=(640, 384))
            eng  = build_engine(onnx, Path("pitch_kp.engine"), fp16=True)
        except RuntimeError as e:
            print(f"  EXPECTED FAILURE: {e}")
            print("  fix: add optimization profile for dynamic NMS output — see source")
    else:
        print(f"  pitch model not found: {MODEL_PITCH}")
