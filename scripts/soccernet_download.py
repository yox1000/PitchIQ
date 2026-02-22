#!/usr/bin/env python3
"""Download SoccerNet games and auto-select 4 high-quality scenario clips.

Outputs clips to:
  data/video/scenarios/
    - press_recovery.mp4
    - through_ball.mp4
    - midfield_buildup.mp4
    - set_piece.mp4
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


SCENARIO_NAMES = ["press_recovery", "through_ball", "midfield_buildup", "set_piece"]


@dataclass
class FrameMetric:
    t: float
    green_ratio: float
    brightness: float
    sharpness: float
    motion: float
    cut_flag: int
    left_box_density: float
    right_box_density: float
    center_density: float


@dataclass
class WindowCandidate:
    video_path: str
    match_id: str
    start_s: float
    duration_s: float
    quality: float
    score_press: float
    score_through: float
    score_midfield: float
    score_set_piece: float
    mean_green: float
    mean_motion: float
    cut_count: int

    def score_for(self, scenario: str) -> float:
        if scenario == "press_recovery":
            return self.score_press
        if scenario == "through_ball":
            return self.score_through
        if scenario == "midfield_buildup":
            return self.score_midfield
        if scenario == "set_piece":
            return self.score_set_piece
        return -1e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default="/workspace/data/video/soccernet", help="SoccerNet local root directory")
    parser.add_argument("--output-dir", default="/workspace/data/video/scenarios", help="Output clip directory")
    parser.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"), help="SoccerNet password")
    parser.add_argument("--split", default="train", help="SoccerNet split")
    parser.add_argument("--download-count", type=int, default=10, help="How many game indexes to download")
    parser.add_argument("--files", nargs="+", default=["1_720p.mkv", "2_720p.mkv"], help="SoccerNet file names")
    parser.add_argument("--no-download", action="store_true", help="Skip downloader and only run clip selection")
    parser.add_argument("--clip-seconds", type=float, default=38.0, help="Clip duration target (35-40s recommended)")
    parser.add_argument("--window-step", type=float, default=2.0, help="Window step in seconds")
    parser.add_argument("--sample-hz", type=float, default=1.0, help="Sampling frequency for quality analysis")
    parser.add_argument(
        "--scan-max-seconds",
        type=float,
        default=2400.0,
        help="Max seconds to scan from each match video for speed (default: 40 min)",
    )
    return parser.parse_args()


def download_soccernet(local_dir: Path, password: str, split: str, files: List[str], count: int) -> None:
    from SoccerNet.Downloader import SoccerNetDownloader, getListGames

    dl = SoccerNetDownloader(LocalDirectory=str(local_dir))
    dl.password = password
    target_count = max(1, int(count))

    # Preferred path for current SoccerNet API: enumerate split games and download first N.
    try:
        games = list(getListGames(split=[split], task="spotting", dataset="SoccerNet"))
    except Exception:
        games = []

    if len(games) > 0 and hasattr(dl, "downloadGame"):
        for game in games[:target_count]:
            try:
                dl.downloadGame(game, files=files, spl=split, verbose=True)
            except TypeError:
                # Some versions don't accept 'spl'
                dl.downloadGame(game, files=files, verbose=True)
        return

    # Fallback compatibility for older/newer APIs.
    sig = inspect.signature(dl.downloadGames)
    params = set(sig.parameters.keys())
    kwargs = {"files": files, "split": [split]}
    if "index" in params:
        kwargs["index"] = list(range(target_count))
    dl.downloadGames(**kwargs)


def list_match_videos(local_dir: Path, files: List[str]) -> List[Path]:
    wanted = set(files)
    vids = [p for p in local_dir.rglob("*") if p.is_file() and p.name in wanted]
    vids = sorted(vids)
    return vids


def _hist_hs(frame_hsv: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([frame_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def _frame_stats(frame_small: np.ndarray, prev_small: np.ndarray | None, prev_hist: np.ndarray | None) -> Tuple[FrameMetric, np.ndarray]:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
    h, w = frame_small.shape[:2]

    green = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 30))
    green_ratio = float(np.mean(green))
    non_green = (~green).astype(np.float32)

    brightness = float(np.mean(hsv[..., 2]))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hist = _hist_hs(hsv)
    if prev_small is None or prev_hist is None:
        motion = 0.0
        cut_flag = 0
    else:
        motion = float(np.mean(cv2.absdiff(frame_small, prev_small)))
        corr = float(cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL))
        cut_flag = int(corr < 0.72 and motion > 14.0)

    left_box = non_green[int(0.20 * h) : int(0.84 * h), 0 : int(0.26 * w)]
    right_box = non_green[int(0.20 * h) : int(0.84 * h), int(0.74 * w) : w]
    center = non_green[int(0.28 * h) : int(0.78 * h), int(0.36 * w) : int(0.64 * w)]

    metric = FrameMetric(
        t=0.0,
        green_ratio=green_ratio,
        brightness=brightness,
        sharpness=sharpness,
        motion=motion,
        cut_flag=cut_flag,
        left_box_density=float(np.mean(left_box)) if left_box.size else 0.0,
        right_box_density=float(np.mean(right_box)) if right_box.size else 0.0,
        center_density=float(np.mean(center)) if center.size else 0.0,
    )
    return metric, hist


def analyze_video(video_path: Path, clip_seconds: float, step_seconds: float, sample_hz: float, scan_max_seconds: float) -> List[WindowCandidate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / max(fps, 1e-6)
    scan_duration = min(float(scan_max_seconds), duration)

    sample_step_frames = max(1, int(round(fps / max(sample_hz, 0.1))))
    frame_metrics: List[FrameMetric] = []
    prev_small = None
    prev_hist = None
    frame_idx = 0
    sample_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_step_frames != 0:
            frame_idx += 1
            continue
        t = frame_idx / max(fps, 1e-6)
        if t > scan_duration:
            break

        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        m, hist = _frame_stats(small, prev_small, prev_hist)
        m.t = float(t)
        frame_metrics.append(m)
        prev_small = small
        prev_hist = hist

        sample_idx += 1
        frame_idx += 1

    cap.release()
    if len(frame_metrics) < int(clip_seconds):
        return []

    ts = np.array([m.t for m in frame_metrics], dtype=np.float32)
    green = np.array([m.green_ratio for m in frame_metrics], dtype=np.float32)
    bright = np.array([m.brightness for m in frame_metrics], dtype=np.float32)
    sharp = np.array([m.sharpness for m in frame_metrics], dtype=np.float32)
    motion = np.array([m.motion for m in frame_metrics], dtype=np.float32)
    cuts = np.array([m.cut_flag for m in frame_metrics], dtype=np.float32)
    left_box = np.array([m.left_box_density for m in frame_metrics], dtype=np.float32)
    right_box = np.array([m.right_box_density for m in frame_metrics], dtype=np.float32)
    center_box = np.array([m.center_density for m in frame_metrics], dtype=np.float32)

    clip_samples = max(4, int(round(clip_seconds * sample_hz)))
    step_samples = max(1, int(round(step_seconds * sample_hz)))
    candidates: List[WindowCandidate] = []

    rel = video_path.relative_to(video_path.parents[2]) if len(video_path.parents) >= 3 else video_path
    match_id = str(rel.parent)

    for s in range(0, len(frame_metrics) - clip_samples, step_samples):
        e = s + clip_samples
        g = float(np.mean(green[s:e]))
        b = float(np.mean(bright[s:e]))
        sh = float(np.mean(sharp[s:e]))
        mm = float(np.mean(motion[s:e]))
        mp95 = float(np.percentile(motion[s:e], 95))
        mv = float(np.std(motion[s:e]))
        cc = int(np.sum(cuts[s:e]))
        lbd = float(np.mean(left_box[s:e]))
        rbd = float(np.mean(right_box[s:e]))
        cbd = float(np.mean(center_box[s:e]))
        box_density = max(lbd, rbd)

        # Strict quality gate for usable clips.
        if not (g >= 0.33 and b >= 45.0 and sh >= 35.0 and cc == 0 and 0.8 <= mm <= 22.0 and mp95 <= 35.0):
            continue

        motion_score = math.exp(-((mm - 7.0) ** 2) / (2.0 * (5.0 ** 2)))
        quality = (2.0 * g) + (0.7 * min(sh / 130.0, 1.0)) + (0.5 * min(b / 130.0, 1.0)) + motion_score

        score_press = quality + (0.18 * mm) + (0.35 * cbd)
        score_through = quality + (0.12 * mp95) + (0.10 * mv)
        score_midfield = quality + (0.9 * max(0.0, 1.0 - abs(mm - 6.0) / 6.0)) + (0.35 * g) - (0.2 * box_density)
        score_set_piece = quality + (0.9 * box_density) + (0.4 * max(0.0, 1.0 - mm / 10.0))

        candidates.append(
            WindowCandidate(
                video_path=str(video_path),
                match_id=match_id,
                start_s=float(ts[s]),
                duration_s=float(clip_seconds),
                quality=float(quality),
                score_press=float(score_press),
                score_through=float(score_through),
                score_midfield=float(score_midfield),
                score_set_piece=float(score_set_piece),
                mean_green=float(g),
                mean_motion=float(mm),
                cut_count=cc,
            )
        )

    return candidates


def pick_four(candidates: List[WindowCandidate]) -> Dict[str, WindowCandidate]:
    chosen: Dict[str, WindowCandidate] = {}
    used_matches: set[str] = set()
    used_windows: Dict[str, List[Tuple[float, float]]] = {}

    for scenario in SCENARIO_NAMES:
        ranked = sorted(candidates, key=lambda c: c.score_for(scenario), reverse=True)
        picked = None
        for c in ranked:
            if c.match_id in used_matches:
                continue
            # Avoid overlap inside same source file.
            blocked = False
            for a, b in used_windows.get(c.video_path, []):
                if not (c.start_s + c.duration_s <= a or c.start_s >= b):
                    blocked = True
                    break
            if blocked:
                continue
            picked = c
            break
        if picked is None and ranked:
            picked = ranked[0]
        if picked is None:
            raise RuntimeError(f"No candidate clip found for scenario: {scenario}")
        chosen[scenario] = picked
        used_matches.add(picked.match_id)
        used_windows.setdefault(picked.video_path, []).append((picked.start_s, picked.start_s + picked.duration_s))

    return chosen


def ffmpeg_cut_copy(src: Path, dst: Path, start_s: float, duration_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "1",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        if not args.password:
            raise ValueError("SoccerNet password is required (pass --password or set SOCCERNET_PASSWORD).")
        print(f"[download] split={args.split} count={args.download_count} -> {local_dir}")
        download_soccernet(
            local_dir=local_dir,
            password=args.password,
            split=args.split,
            files=list(args.files),
            count=int(args.download_count),
        )

    videos = list_match_videos(local_dir, list(args.files))
    if len(videos) == 0:
        raise FileNotFoundError(f"No SoccerNet videos found in: {local_dir}")
    print(f"[scan] found {len(videos)} candidate source videos")

    all_candidates: List[WindowCandidate] = []
    for i, vp in enumerate(videos, 1):
        print(f"[scan] {i}/{len(videos)} {vp}")
        cands = analyze_video(
            video_path=vp,
            clip_seconds=float(args.clip_seconds),
            step_seconds=float(args.window_step),
            sample_hz=float(args.sample_hz),
            scan_max_seconds=float(args.scan_max_seconds),
        )
        print(f"       candidates={len(cands)}")
        all_candidates.extend(cands)

    if len(all_candidates) == 0:
        raise RuntimeError("No windows passed quality gates. Increase --scan-max-seconds or relax thresholds in script.")

    chosen = pick_four(all_candidates)
    manifest = {"clips": {}, "source_video_count": len(videos), "candidate_count": len(all_candidates)}

    for scenario in SCENARIO_NAMES:
        c = chosen[scenario]
        src = Path(c.video_path)
        dst = output_dir / f"{scenario}.mp4"
        ffmpeg_cut_copy(src=src, dst=dst, start_s=c.start_s, duration_s=c.duration_s)
        print(f"[clip] {scenario}: {src} @ {c.start_s:.1f}s -> {dst}")
        manifest["clips"][scenario] = {
            "src": str(src),
            "match_id": c.match_id,
            "start_s": round(float(c.start_s), 3),
            "duration_s": round(float(c.duration_s), 3),
            "quality": round(float(c.quality), 4),
            "score_press": round(float(c.score_press), 4),
            "score_through": round(float(c.score_through), 4),
            "score_midfield": round(float(c.score_midfield), 4),
            "score_set_piece": round(float(c.score_set_piece), 4),
            "mean_green": round(float(c.mean_green), 4),
            "mean_motion": round(float(c.mean_motion), 4),
            "cut_count": int(c.cut_count),
            "out_clip": str(dst),
        }

    manifest_path = output_dir / "soccernet_clip_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[done] wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
