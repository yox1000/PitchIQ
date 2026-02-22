#!/usr/bin/env python3.10
"""Run bottlejob pipeline on multiple 30s scenarios and export judge artifacts."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

try:
    from scripts.runtime_overrides import apply_runtime_overrides
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_overrides import apply_runtime_overrides


NOTEBOOK_PATH = Path("notebooks/bottlejob_v2.ipynb")
EXEC_CELLS = [3, 5, 9, 10, 11, 12, 14, 16, 28, 29, 30]

SCENARIOS = [
    {"name": "bruno", "goal_times_s": []},
    {"name": "martial", "goal_times_s": []},
    {"name": "giroud", "goal_times_s": []},
    {"name": "scenario5", "goal_times_s": []},
    {"name": "scenario2", "goal_times_s": []},
    {"name": "scenario3", "goal_times_s": []},
    {"name": "scenario4", "goal_times_s": []},
    {"name": "press_recovery", "goal_times_s": []},
    {"name": "through_ball", "goal_times_s": []},
    {"name": "midfield_buildup", "goal_times_s": []},
    {"name": "set_piece", "goal_times_s": []},
]

MODEL_GDRIVE_IDS = {
    "football-player-detection.pt": "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q",
    "football-pitch-detection.pt": "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name to run (repeatable). Default runs all scenarios.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=2,
        help="Frame decimation factor used by process_video_multi_output (default: 2).",
    )
    parser.add_argument(
        "--no-overrides",
        action="store_true",
        help="Disable runtime overrides and run notebook logic as-is.",
    )
    parser.add_argument(
        "--mode",
        choices=["tracking_only", "pass_optimality", "eval_bar", "tactical"],
        default=None,
        help="Single render mode. Omit --mode to render all 4 modes.",
    )
    return parser.parse_args()


def ensure_model_weights(model_dir: Path) -> None:
    """Download required model weights if missing."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, file_id in MODEL_GDRIVE_IDS.items():
        out_path = model_dir / filename
        if out_path.exists():
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[models] missing {filename}, downloading from Google Drive...")
        cmd = ["python", "-m", "gdown", url, "-O", str(out_path)]
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0 or (not out_path.exists()):
            raise RuntimeError(f"Failed to download required model: {filename}")


def resolve_scenario_video(name: str) -> Path:
    for base in (Path("/workspace/data/video/scenarios"), Path("data/video/scenarios")):
        p = base / f"{name}.mp4"
        if p.exists():
            return p
    # default to requested workspace path for clearer missing-file message
    return Path("/workspace/data/video/scenarios") / f"{name}.mp4"


def load_notebook_context(notebook_path: Path) -> dict:
    nb = nbformat.read(notebook_path, as_version=4)
    ctx: dict = {}
    for idx in EXEC_CELLS:
        exec(compile(nb.cells[idx].source, f"<cell_{idx}>", "exec"), ctx)
    return ctx


def get_video_meta(video_path: Path) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (frames / fps) if fps > 0 else 0.0
    return {"fps": fps, "frames": frames, "duration_s": duration, "width": width, "height": height}


def render_tracking_only(
    video_path: Path,
    out_path: Path,
    analyzer,
    skip_frames: int,
    max_frames: int | None = None,
    start_frame: int = 0,
) -> Dict[str, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_frame = max(0, int(start_frame))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    total_frames = max(0, total_frames_all - start_frame)
    if max_frames is not None:
        total_frames = min(total_frames, int(max_frames))

    out_fps = fps / max(1, int(skip_frames))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))

    frame_idx = 0
    processed = 0
    dot_radius = max(6, min(10, int(round(height / 90.0))))

    while frame_idx < total_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, int(skip_frames)) != 0:
            frame_idx += 1
            continue

        result = analyzer.process_frame(frame)
        vis = frame.copy()
        det = result.get("detections")
        teams = result.get("teams")
        if det is not None and teams is not None and len(det) > 0:
            teams_arr = np.asarray(teams, dtype=int)
            for i, (xyxy, cls_id) in enumerate(zip(det.xyxy, det.class_id)):
                # Only team players + goalkeepers with known team labels.
                if cls_id not in (analyzer.cfg.PLAYER, analyzer.cfg.GOALKEEPER):
                    continue
                team = int(teams_arr[i]) if i < len(teams_arr) else -1
                if team not in (0, 1):
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, y2
                color = (128, 0, 255) if team == 0 else (0, 215, 255)  # purple vs yellow/gold
                cv2.circle(vis, (cx, cy), dot_radius, color, -1)
                cv2.circle(vis, (cx, cy), dot_radius, (0, 0, 0), 2)

        writer.write(vis)
        processed += 1
        frame_idx += 1

    cap.release()
    writer.release()
    return get_video_meta(out_path)


def save_eval_csv_and_plot(eval_history: List[float], output_fps: float, out_csv: Path, out_plot: Path) -> None:
    times = np.arange(len(eval_history), dtype=float) / max(output_fps, 1e-6)
    df = pd.DataFrame({"time_s": times, "eval_bar": np.asarray(eval_history, dtype=float)})
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.plot(df["time_s"], df["eval_bar"], color="#007f5f", linewidth=2)
    ax.set_title(f"Eval Bar vs Time ({out_csv.stem})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Eval")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_plot)
    plt.close(fig)


def draw_reprojection_frame(frame: np.ndarray, result: dict, out_path: Path) -> bool:
    src = result.get("homography_source_pts")
    reproj = result.get("homography_reprojected_pts")
    err = result.get("homography_reprojection_error_px")
    if src is None or reproj is None or err is None:
        return False

    vis = frame.copy()
    for (sx, sy), (rx, ry) in zip(src, reproj):
        cv2.circle(vis, (int(round(sx)), int(round(sy))), 5, (0, 0, 255), -1)      # red = source
        cv2.circle(vis, (int(round(rx)), int(round(ry))), 5, (0, 255, 0), 2)        # green = reprojected
    cv2.putText(
        vis,
        f"Mean reprojection error: {float(err):.2f}px",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return True


def collect_sample_metrics(
    video_path: Path,
    analyzer_factory,
    stride: int = 10,
    max_samples: int = 120,
    reprojection_out: Path | None = None,
) -> Dict[str, float]:
    analyzer = analyzer_factory()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for metrics: {video_path}")

    sample_idx = 0
    frame_idx = 0
    homography_errors: List[float] = []
    player_counts: List[int] = []
    unknown_team_ratios: List[float] = []
    used_reprojection = False

    while sample_idx < max_samples:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        result = analyzer.process_frame(frame)
        det = result.get("detections")
        teams = result.get("teams")
        if det is not None and teams is not None:
            player_mask = (det.class_id == analyzer.cfg.PLAYER) | (det.class_id == analyzer.cfg.GOALKEEPER)
            n_players = int(player_mask.sum())
            player_counts.append(n_players)
            if n_players > 0:
                unknown = int((teams[player_mask] == -1).sum())
                unknown_team_ratios.append(float(unknown) / float(n_players))

        err = result.get("homography_reprojection_error_px")
        if err is not None and np.isfinite(err):
            homography_errors.append(float(err))
            if reprojection_out is not None and (not used_reprojection):
                used_reprojection = draw_reprojection_frame(frame, result, reprojection_out)

        sample_idx += 1
        frame_idx += 1

    cap.release()

    return {
        "samples_used": float(sample_idx),
        "homography_mean_px": float(np.mean(homography_errors)) if homography_errors else float("nan"),
        "homography_p90_px": float(np.percentile(homography_errors, 90)) if homography_errors else float("nan"),
        "player_count_mean": float(np.mean(player_counts)) if player_counts else float("nan"),
        "unknown_team_ratio_mean": float(np.mean(unknown_team_ratios)) if unknown_team_ratios else float("nan"),
    }


def eval_spike_stats(eval_csv: Path, goal_times_s: List[float], spike_threshold: float = 12.0) -> Dict[str, object]:
    df = pd.read_csv(eval_csv)
    t_col = "time_s" if "time_s" in df.columns else df.columns[0]
    e_col = "eval_bar" if "eval_bar" in df.columns else df.columns[-1]

    abs_delta = df[e_col].diff().abs().fillna(0.0)
    spikes = df.loc[abs_delta > spike_threshold, t_col].to_numpy(dtype=float)

    goals_with_lead_spike = 0
    for goal_t in goal_times_s:
        if np.any((spikes >= (goal_t - 5.0)) & (spikes < goal_t)):
            goals_with_lead_spike += 1

    return {
        "spike_threshold": spike_threshold,
        "spike_times_s": [round(float(v), 2) for v in spikes.tolist()],
        "goal_times_s": [round(float(v), 2) for v in goal_times_s],
        "spiked_before_goals_n": int(goals_with_lead_spike),
        "goals_n": int(len(goal_times_s)),
        "eval_min": float(df[e_col].min()),
        "eval_max": float(df[e_col].max()),
        "eval_mean": float(df[e_col].mean()),
    }


def main() -> None:
    args = parse_args()
    ctx = load_notebook_context(NOTEBOOK_PATH)
    if not args.no_overrides:
        ctx = apply_runtime_overrides(ctx)
    Config = ctx["Config"]
    YOLO = ctx["YOLO"]
    SoccerAnalyzer = ctx["SoccerAnalyzer"]
    process_video_multi_output = ctx["process_video_multi_output"]

    cfg = Config()
    cfg.player_model = "data/models/football-player-detection.pt"
    cfg.pitch_model = "data/models/football-pitch-detection.pt"
    cfg.output_path = "outputs/"
    ensure_model_weights(Path("data/models"))

    player_model = YOLO(cfg.player_model)
    pitch_model = YOLO(cfg.pitch_model)

    out_root = Path("outputs/scenario_suite")
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {"scenarios": [], "mode": args.mode or "all"}

    all_specs = list(SCENARIOS)
    selected = set(args.scenario or [])
    scenario_specs = [s for s in all_specs if not selected or s["name"] in selected]
    if len(scenario_specs) == 0:
        valid = ", ".join(s["name"] for s in all_specs)
        raise ValueError(f"No matching scenarios found for --scenario. Valid names: {valid}")

    mode = args.mode
    want_tracking = mode in (None, "tracking_only")
    want_pass = mode in (None, "pass_optimality")
    want_eval = mode in (None, "eval_bar")
    want_tactical = mode in (None, "tactical")
    run_pipeline = want_pass or want_eval or want_tactical

    for spec in scenario_specs:
        name = spec["name"]
        video_path = resolve_scenario_video(name)
        if not video_path.exists():
            raise FileNotFoundError(f"Scenario video missing: {video_path}")

        scenario_dir = out_root / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        # Internal renderer still emits legacy names; keep them in a temp dir and clean up.
        judge_dir = scenario_dir / "_tmp_render"
        if judge_dir.exists():
            shutil.rmtree(judge_dir)

        meta_in = get_video_meta(video_path)
        print(f"\\n=== {name} ===")
        print(f"input: {video_path} | {meta_in['duration_s']:.2f}s @ {meta_in['fps']:.3f} fps")

        eval_history: List[float] = []
        if run_pipeline:
            analyzer = SoccerAnalyzer(player_model, pitch_model, cfg)
            setattr(analyzer, "_rt_scenario_name", str(name))
            eval_history = process_video_multi_output(
                str(video_path),
                str(judge_dir),
                analyzer,
                max_frames=int(meta_in["frames"]),
                skip_frames=max(1, int(args.skip_frames)),
                start_frame=0,
            )

        out_meta: Dict[str, Dict[str, float]] = {}
        outputs: Dict[str, str] = {}

        if run_pipeline:
            overlay_src = judge_dir / "eval_overlay.mp4"
            pass_src = judge_dir / "pass_prediction.mp4"
            tactical_src = judge_dir / "tactical_view.mp4"

            overlay_out = scenario_dir / "eval_bar.mp4"
            pass_out = scenario_dir / "pass_optimality.mp4"
            tactical_out = scenario_dir / "tactical.mp4"

            if want_eval:
                shutil.copy2(overlay_src, overlay_out)
                outputs["eval_bar"] = str(overlay_out)
                out_meta["eval_bar"] = get_video_meta(overlay_out)
            if want_pass:
                shutil.copy2(pass_src, pass_out)
                outputs["pass_optimality"] = str(pass_out)
                out_meta["pass_optimality"] = get_video_meta(pass_out)
            if want_tactical:
                shutil.copy2(tactical_src, tactical_out)
                outputs["tactical"] = str(tactical_out)
                out_meta["tactical"] = get_video_meta(tactical_out)

        tracking_out = scenario_dir / "tracking_only.mp4"
        if want_tracking:
            tracking_analyzer = SoccerAnalyzer(player_model, pitch_model, cfg)
            setattr(tracking_analyzer, "_rt_scenario_name", str(name))
            out_meta["tracking_only"] = render_tracking_only(
                video_path=video_path,
                out_path=tracking_out,
                analyzer=tracking_analyzer,
                skip_frames=max(1, int(args.skip_frames)),
                max_frames=int(meta_in["frames"]),
                start_frame=0,
            )
            outputs["tracking_only"] = str(tracking_out)

        eval_csv = scenario_dir / f"{name}_eval_ts.csv"
        eval_plot = scenario_dir / f"{name}_eval_plot.png"
        if run_pipeline and len(eval_history) > 0:
            output_fps = float(meta_in["fps"]) / max(1, int(args.skip_frames))
            save_eval_csv_and_plot(eval_history, output_fps, eval_csv, eval_plot)
        # Remove internal legacy 3-video outputs so only the 4 requested mode videos remain.
        if judge_dir.exists():
            shutil.rmtree(judge_dir, ignore_errors=True)

        # Independent sampled metrics (isolated pass)
        def analyzer_factory():
            return SoccerAnalyzer(player_model, pitch_model, cfg)

        reprojection_png = scenario_dir / f"{name}_reprojection.png"
        sampled = collect_sample_metrics(
            video_path,
            analyzer_factory=analyzer_factory,
            stride=20,
            max_samples=36,
            reprojection_out=reprojection_png,
        )
        spike = eval_spike_stats(eval_csv, goal_times_s=spec.get("goal_times_s", [])) if (run_pipeline and eval_csv.exists()) else {}

        scenario_summary = {
            "name": name,
            "input_video": str(video_path),
            "input_meta": meta_in,
            "outputs": {**outputs, "eval_csv": str(eval_csv), "eval_plot": str(eval_plot), "reprojection_png": str(reprojection_png)},
            "output_meta": out_meta,
            "sampled_metrics": sampled,
            "spike_metrics": spike,
        }
        summary["scenarios"].append(scenario_summary)

        with (scenario_dir / f"{name}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(scenario_summary, f, indent=2)

        print(f"[final outputs] {scenario_dir}")
        for mode_name in ("tracking_only", "pass_optimality", "eval_bar", "tactical"):
            if mode_name in outputs:
                print(f"  - {outputs[mode_name]}")

    summary_path = out_root / "scenario_suite_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
