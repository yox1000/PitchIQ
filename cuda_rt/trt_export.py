"""
tensorrt export attempt — unfinished

plan was to export both yolo models to trt fp16 engines
and use them via tensorrt python api instead of ultralytics.

got player detection working, pitch keypoint model has dynamic shapes
that trt doesn't like without explicit optimization profiles.
ran out of time.
"""

import torch
import numpy as np
from pathlib import Path

MODEL_PLAYER = Path("../data/models/football-player-detection.pt")
MODEL_PITCH  = Path("../data/models/football-pitch-detection.pt")


def export_onnx(model_path, out_path, img_size=(640, 384)):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    # ultralytics has a built-in export but it doesn't set the right
    # dynamic axes for variable batch size. doing it manually.
    model.export(format="onnx", imgsz=img_size, dynamic=False, simplify=True)
    print(f"exported {out_path}")


def build_trt_engine(onnx_path, engine_path, fp16=True, workspace_gb=4):
    try:
        import tensorrt as trt
    except ImportError:
        print("tensorrt not installed — pip install tensorrt")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_gb * (1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("fp16 enabled")

    # this is where the pitch keypoint model fails —
    # it has a dynamic output shape for keypoints that trt doesn't resolve
    # without an explicit optimization profile. TODO: add profiles.
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("trt engine build failed")
        return None

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    print(f"engine saved: {engine_path}")
    return engine_path


def trt_infer(engine_path, img_np):
    """
    run inference on a pre-built trt engine.
    img_np: uint8 HWC BGR, will be normalised to fp16 NCHW
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    ctx = engine.create_execution_context()

    # pre-process
    img = img_np[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, norm
    img = np.transpose(img, (2, 0, 1))[None]              # NCHW
    img = np.ascontiguousarray(img, dtype=np.float16)

    # alloc buffers
    in_size  = int(np.prod(img.shape)) * 2   # fp16 = 2 bytes
    out_size = 1 * 84 * 8400 * 2            # yolov8 default output shape fp16

    d_in  = cuda.mem_alloc(in_size)
    d_out = cuda.mem_alloc(out_size)
    h_out = np.empty((1, 84, 8400), dtype=np.float16)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_in, img, stream)
    ctx.execute_async_v2([int(d_in), int(d_out)], stream.handle)
    cuda.memcpy_dtoh_async(h_out, d_out, stream)
    stream.synchronize()

    return h_out


if __name__ == "__main__":
    # step 1: export onnx
    if MODEL_PLAYER.exists():
        export_onnx(MODEL_PLAYER, "player_det.onnx")
    else:
        print(f"model not found: {MODEL_PLAYER}")
        print("run: python -m gdown ... (see README)")

    # step 2: build trt engine
    # build_trt_engine("player_det.onnx", "player_det.engine")
    # ^ commented out — pitch keypoint model build_trt fails, fixing profiles first
