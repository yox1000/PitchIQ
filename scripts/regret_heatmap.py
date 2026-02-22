#!/usr/bin/env python3
"""Compute soccer pass-regret heatmaps and render match analysis video."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from mplsoccer import Pitch
except Exception:
    Pitch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.notebook_runtime import build_runtime
from scripts.run_scenario_suite import ensure_model_weights


GRID_X = 16
GRID_Y = 10
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# User-provided halftime timing for this match.
HALFTIME_START_S = 2902.0  # 48:22
SECOND_HALF_START_S = 2943.0  # 49:03


@dataclass
class FrameState:
    frame_idx: int
    timestamp_s: float
    valid: bool
    holder_team: int
    holder_track_id: Optional[int]
    holder_is_goalkeeper: bool
    holder_pitch: Optional[np.ndarray]
    holder_pixel: Optional[Tuple[int, int]]
    ball_pitch: Optional[np.ndarray]
    ball_detected: bool
    options: List[Dict[str, Any]]
    skip_reason: str


@dataclass
class BallMemory:
    last_pixel: Optional[np.ndarray]
    last_pitch: Optional[np.ndarray]
    vel_pixel: np.ndarray
    vel_pitch: np.ndarray
    miss_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input full-match video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--team",
        type=int,
        choices=[0, 1],
        default=None,
        help="Deprecated compatibility arg; both team heatmaps are always generated.",
    )
    parser.add_argument("--skip-frames", type=int, default=3, help="Frame stride (default: 3)")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional raw-frame cap for debugging")
    return parser.parse_args()


def _to_track_id(raw_id: Any) -> Optional[int]:
    if raw_id is None:
        return None
    try:
        if np.isnan(raw_id):
            return None
    except Exception:
        pass
    try:
        return int(raw_id)
    except Exception:
        return None


def _cell_index(x: float, y: float) -> Tuple[int, int]:
    x = float(np.clip(x, 0.0, PITCH_LENGTH))
    y = float(np.clip(y, 0.0, PITCH_WIDTH))
    ix = min(int((x / PITCH_LENGTH) * GRID_X), GRID_X - 1)
    iy = min(int((y / PITCH_WIDTH) * GRID_Y), GRID_Y - 1)
    return ix, iy


def _team_relative_xt(get_xt_fn: Any, x: float, y: float, team_id: int) -> float:
    xx = float(np.clip(x, 0.0, PITCH_LENGTH))
    # Team-relative xT lookup:
    # - team 0 uses base grid (high on right)
    # - team 1 uses horizontally flipped grid (high on left)
    if int(team_id) == 1:
        xx = float(PITCH_LENGTH) - xx
    yy = float(np.clip(y, 0.0, PITCH_WIDTH))
    return float(get_xt_fn(xx, yy))


def _estimate_player_voronoi_areas(positions: np.ndarray, nx: int = 42, ny: int = 28) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float32)
    if len(pos) == 0:
        return np.zeros((0,), dtype=np.float32)
    xs = np.linspace(0.0, PITCH_LENGTH, int(nx), dtype=np.float32)
    ys = np.linspace(0.0, PITCH_WIDTH, int(ny), dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    d = np.linalg.norm(grid[:, None, :] - pos[None, :, :], axis=2)
    owner = np.argmin(d, axis=1)
    counts = np.bincount(owner, minlength=len(pos)).astype(np.float32)
    cell_area = (PITCH_LENGTH / float(nx)) * (PITCH_WIDTH / float(ny))
    return counts * float(cell_area)


def _compute_permissive_options(
    holder_pos: np.ndarray,
    holder_team: int,
    player_pitch: np.ndarray,
    player_teams: np.ndarray,
    find_pass_options_fn: Any,
    get_xt_fn: Any,
    xt_min: float,
    xt_max: float,
) -> List[Dict[str, Any]]:
    raw_opts = find_pass_options_fn(holder_pos, int(holder_team), player_pitch, player_teams)
    if len(raw_opts) == 0:
        return []

    areas = _estimate_player_voronoi_areas(player_pitch)
    mates = np.where(player_teams == int(holder_team))[0]
    mate_areas = areas[mates] if len(mates) > 0 else np.zeros((0,), dtype=np.float32)
    a_min = float(np.min(mate_areas)) if len(mate_areas) > 0 else 0.0
    a_max = float(np.max(mate_areas)) if len(mate_areas) > 0 else 1.0
    a_den = max(a_max - a_min, 1e-6)
    xt_den = max(float(xt_max) - float(xt_min), 1e-6)

    best_idx = -1
    best_score = -1.0
    for i, opt in enumerate(raw_opts):
        t = np.asarray(opt.get("target_pos"), dtype=np.float32)
        if t.shape != (2,):
            opt["optimal_score"] = 0.0
            opt["is_best"] = False
            continue
        d = np.linalg.norm(player_pitch - t[None, :], axis=1)
        ridx = int(np.argmin(d))
        rec_area = float(areas[ridx]) if ridx < len(areas) else 0.0
        area_norm = float(np.clip((rec_area - a_min) / a_den, 0.0, 1.0))
        end_xt = _team_relative_xt(get_xt_fn, float(t[0]), float(t[1]), int(holder_team))
        xt_norm = float(np.clip((end_xt - float(xt_min)) / xt_den, 0.0, 1.0))
        score = float((0.70 * xt_norm) + (0.30 * area_norm))
        opt["optimal_score"] = float(np.clip(score, 0.0, 1.0))
        opt["is_best"] = False
        if opt["optimal_score"] > best_score:
            best_score = float(opt["optimal_score"])
            best_idx = i
    if best_idx >= 0:
        raw_opts[best_idx]["is_best"] = True
    return raw_opts


def _extract_state(
    result: Dict[str, Any],
    analyzer: Any,
    frame_idx: int,
    timestamp_s: float,
    frame_shape: Tuple[int, int],
    ball_memory: BallMemory,
    runtime_ctx: Dict[str, Any],
) -> FrameState:
    det = result.get("detections")
    pitch_coords = result.get("pitch_coords")
    if det is None or pitch_coords is None or len(det) == 0:
        return FrameState(frame_idx, float(timestamp_s), False, -1, None, False, None, None, None, False, [], "no_detections")

    class_ids = np.asarray(det.class_id)
    player_mask = (class_ids == analyzer.cfg.PLAYER) | (class_ids == analyzer.cfg.GOALKEEPER)
    player_idxs = np.where(player_mask)[0]
    if len(player_idxs) == 0:
        return FrameState(frame_idx, float(timestamp_s), False, -1, None, False, None, None, None, False, [], "no_players")

    player_pixels = det.get_anchors_coordinates(anchor=analyzer.ctx_sv.Position.BOTTOM_CENTER)[player_mask]
    player_pitch = np.asarray(pitch_coords[player_mask], dtype=np.float32)
    teams_full = np.asarray(result.get("teams"), dtype=int) if result.get("teams") is not None else np.full((len(det),), -1, dtype=int)
    player_teams = teams_full[player_mask]

    # Ball position: runtime -> permissive detector fallback -> short interpolation.
    h, w = int(frame_shape[0]), int(frame_shape[1])
    ball_detected = False
    ball_interpolated = False
    ball_px_vec: Optional[np.ndarray] = None
    ball_pitch: Optional[np.ndarray] = None

    if bool(result.get("_rt_ball_detected", False)) and result.get("_rt_ball_pixel") is not None and result.get("ball_pos") is not None:
        ball_px_vec = np.asarray(result.get("_rt_ball_pixel"), dtype=np.float32).reshape(2)
        ball_pitch = np.asarray(result.get("ball_pos"), dtype=np.float32).reshape(2)
        ball_detected = True
    else:
        ball_idx = np.where(class_ids == analyzer.cfg.BALL)[0]
        if len(ball_idx) > 0:
            if getattr(det, "confidence", None) is not None:
                conf = np.asarray(det.confidence, dtype=np.float32)
                good = [int(i) for i in ball_idx if float(conf[int(i)]) >= 0.03]
            else:
                conf = np.ones((len(det),), dtype=np.float32)
                good = [int(i) for i in ball_idx]

            if len(good) > 0:
                last_px = ball_memory.last_pixel
                search_radius = 220.0
                candidate_idxs = good
                if last_px is not None:
                    dists = []
                    for i in candidate_idxs:
                        x1, y1, x2, y2 = [float(v) for v in det.xyxy[int(i)]]
                        c = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
                        dists.append(float(np.linalg.norm(c - last_px)))
                    nearby = [candidate_idxs[k] for k, d in enumerate(dists) if d <= search_radius]
                    if len(nearby) > 0:
                        candidate_idxs = nearby

                pick = int(max(candidate_idxs, key=lambda i: float(conf[int(i)])))
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[pick]]
                ball_px_vec = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
                ball_pitch = np.asarray(pitch_coords[pick], dtype=np.float32).reshape(2)
                ball_detected = True

    if (not ball_detected) and (ball_memory.last_pixel is not None) and (ball_memory.last_pitch is not None) and (ball_memory.miss_frames < 10):
        ball_px_vec = ball_memory.last_pixel + ball_memory.vel_pixel
        ball_pitch = ball_memory.last_pitch + ball_memory.vel_pitch
        ball_px_vec[0] = float(np.clip(ball_px_vec[0], 0.0, max(0, w - 1)))
        ball_px_vec[1] = float(np.clip(ball_px_vec[1], 0.0, max(0, h - 1)))
        ball_pitch[0] = float(np.clip(ball_pitch[0], 0.0, PITCH_LENGTH))
        ball_pitch[1] = float(np.clip(ball_pitch[1], 0.0, PITCH_WIDTH))
        ball_detected = True
        ball_interpolated = True

    # Update ball memory.
    if ball_detected and (ball_px_vec is not None) and (ball_pitch is not None):
        if ball_memory.last_pixel is not None:
            delta_px = ball_px_vec - ball_memory.last_pixel
            ball_memory.vel_pixel = (0.7 * ball_memory.vel_pixel + 0.3 * delta_px).astype(np.float32)
        if ball_memory.last_pitch is not None:
            delta_pitch = ball_pitch - ball_memory.last_pitch
            ball_memory.vel_pitch = (0.7 * ball_memory.vel_pitch + 0.3 * delta_pitch).astype(np.float32)
        ball_memory.last_pixel = ball_px_vec.astype(np.float32)
        ball_memory.last_pitch = ball_pitch.astype(np.float32)
        ball_memory.miss_frames = ball_memory.miss_frames + 1 if ball_interpolated else 0
    else:
        ball_memory.miss_frames += 1

    holder_local = -1
    holder_track_id = None
    holder_pitch = None
    holder_pixel = None
    holder_team = -1
    holder_is_goalkeeper = False
    skip_reason = ""

    # Strict holder assignment from current ball point only.
    if ball_detected and ball_px_vec is not None:
        bpx = np.asarray(ball_px_vec, dtype=np.float32)
        player_boxes = np.asarray(det.xyxy[player_idxs], dtype=np.float32)
        contained: List[int] = []
        for i, (x1, y1, x2, y2) in enumerate(player_boxes):
            pad = 4.0
            if (x1 - pad) <= bpx[0] <= (x2 + pad) and (y1 - pad) <= bpx[1] <= (y2 + pad):
                contained.append(i)
        if len(contained) > 0:
            d = np.linalg.norm(player_pixels[np.asarray(contained)] - bpx[None, :], axis=1)
            k = int(np.argmin(d))
            if float(d[k]) <= 35.0:
                holder_local = int(contained[k])
        else:
            d = np.linalg.norm(player_pixels - bpx[None, :], axis=1)
            if len(d) > 0:
                d_sorted = np.sort(d)
                k = int(np.argmin(d))
                second_gap = float(d_sorted[1] - d_sorted[0]) if len(d_sorted) > 1 else 99.0
                if float(d[k]) <= 32.0 and second_gap >= 6.0:
                    holder_local = k

    if holder_local >= 0:
        det_idx = int(player_idxs[holder_local])
        holder_pitch = np.asarray(player_pitch[holder_local], dtype=np.float32)
        px = player_pixels[holder_local]
        holder_pixel = (int(px[0]), int(px[1]))
        holder_team = int(player_teams[holder_local]) if holder_local < len(player_teams) else -1
        holder_is_goalkeeper = bool(int(class_ids[det_idx]) == int(analyzer.cfg.GOALKEEPER))
        tracker_ids = getattr(det, "tracker_id", None)
        if tracker_ids is not None and len(tracker_ids) > det_idx:
            holder_track_id = _to_track_id(tracker_ids[det_idx])

    if not ball_detected:
        skip_reason = "no_ball"
    elif holder_local < 0:
        skip_reason = "no_holder"
    elif holder_team not in (0, 1):
        skip_reason = "unknown_team"
    elif holder_is_goalkeeper:
        skip_reason = "goalkeeper_holder"

    options = list(result.get("pass_options", []) or [])
    if holder_local >= 0 and holder_team in (0, 1) and len(options) == 0 and holder_pitch is not None:
        get_xt_fn = runtime_ctx["get_xt"]
        find_pass_options_fn = runtime_ctx["find_pass_options"]
        xt_min = float(runtime_ctx.get("XT_MIN", 0.006))
        xt_max = float(runtime_ctx.get("XT_MAX", 0.124))
        options = _compute_permissive_options(
            holder_pos=holder_pitch,
            holder_team=holder_team,
            player_pitch=player_pitch,
            player_teams=player_teams,
            find_pass_options_fn=find_pass_options_fn,
            get_xt_fn=get_xt_fn,
            xt_min=xt_min,
            xt_max=xt_max,
        )

    valid = bool(ball_detected and holder_local >= 0 and holder_team in (0, 1) and (not holder_is_goalkeeper))
    return FrameState(
        frame_idx=frame_idx,
        timestamp_s=float(timestamp_s),
        valid=valid,
        holder_team=holder_team,
        holder_track_id=holder_track_id,
        holder_is_goalkeeper=holder_is_goalkeeper,
        holder_pitch=holder_pitch,
        holder_pixel=holder_pixel,
        ball_pitch=ball_pitch,
        ball_detected=ball_detected,
        options=options,
        skip_reason=skip_reason,
    )


def _find_actual_score(prev_options: List[Dict[str, Any]], receiver_pitch: np.ndarray, max_dist_m: float = 10.0) -> float:
    if len(prev_options) == 0:
        return 0.0
    best_d = 1e9
    score = 0.0
    for opt in prev_options:
        target = np.asarray(opt.get("target_pos"), dtype=np.float32)
        if target.shape != (2,):
            continue
        d = float(np.linalg.norm(target - receiver_pitch))
        if d < best_d:
            best_d = d
            score = float(opt.get("optimal_score", 0.0))
    if best_d > max_dist_m:
        return 0.0
    return float(np.clip(score, 0.0, 1.0))


def _detect_pass_regret_event(prev: Optional[FrameState], cur: FrameState, team_id: int, last_event_idx: int) -> Optional[Dict[str, Any]]:
    if prev is None or not prev.valid:
        return None
    if prev.holder_team != int(team_id):
        return None
    if prev.holder_is_goalkeeper or cur.holder_is_goalkeeper:
        return None
    if len(prev.options) == 0:
        return None
    if (cur.frame_idx - last_event_idx) <= 1:
        return None

    changed_holder = (cur.holder_track_id != prev.holder_track_id) or (cur.holder_team != prev.holder_team)
    if not changed_holder:
        return None

    optimal_score = max(float(o.get("optimal_score", 0.0)) for o in prev.options)
    optimal_score = float(np.clip(optimal_score, 0.0, 1.0))

    same_team_transfer = bool(cur.valid and cur.holder_team == prev.holder_team and cur.holder_pitch is not None)
    if same_team_transfer:
        receiver_shift = float(np.linalg.norm(cur.holder_pitch - prev.holder_pitch)) if prev.holder_pitch is not None else 0.0
        if receiver_shift < 2.0:
            return None
        actual_score = _find_actual_score(prev.options, cur.holder_pitch)
    else:
        actual_score = 0.0

    ball_move = 0.0
    if prev.ball_pitch is not None and cur.ball_pitch is not None:
        ball_move = float(np.linalg.norm(cur.ball_pitch - prev.ball_pitch))
    if ball_move < 1.5 and same_team_transfer:
        return None

    if prev.holder_pitch is None or prev.holder_pixel is None:
        return None
    regret = float(max(0.0, optimal_score - actual_score))
    return {
        "frame_idx": int(cur.frame_idx),
        "timestamp_s": float(prev.timestamp_s),
        "team": int(prev.holder_team),
        "from_pitch": prev.holder_pitch.copy(),
        "from_pixel": tuple(prev.holder_pixel),
        "optimal_score": float(optimal_score),
        "actual_score": float(np.clip(actual_score, 0.0, 1.0)),
        "regret": regret,
    }


def _risk_color(opt: Dict[str, Any], is_worst: bool) -> Tuple[int, int, int]:
    if bool(opt.get("is_best", False)):
        return (0, 255, 0)
    if is_worst:
        return (0, 0, 255)
    return (0, 255, 255)


def _draw_match_frame(
    frame: np.ndarray,
    result: Dict[str, Any],
    state: FrameState,
    analyzer: Any,
    team_colors: Dict[int, Tuple[int, int, int]],
    ball_color: Tuple[int, int, int],
    active_regret_labels: List[Dict[str, Any]],
    frame_idx: int,
) -> np.ndarray:
    out = frame.copy()
    det = result.get("detections")
    if det is not None and len(det) > 0:
        teams = result["teams"] if result.get("teams") is not None else np.full(len(det), -1)
        pixels = det.get_anchors_coordinates(anchor=analyzer.ctx_sv.Position.BOTTOM_CENTER)
        class_ids = np.asarray(det.class_id)

        for i, cls_id in enumerate(class_ids):
            x1, y1, x2, y2 = map(int, det.xyxy[i])
            cx, cy = (x1 + x2) // 2, y2
            if cls_id == analyzer.cfg.BALL:
                cv2.circle(out, (cx, cy), 9, ball_color, -1)
                cv2.circle(out, (cx, cy), 9, (0, 0, 0), 2)
            elif cls_id in (analyzer.cfg.PLAYER, analyzer.cfg.GOALKEEPER):
                color = team_colors.get(int(teams[i]), team_colors[-1])
                cv2.circle(out, (cx, cy), 14, color, -1)
                cv2.circle(out, (cx, cy), 14, (0, 0, 0), 2)

        player_mask = (class_ids == analyzer.cfg.PLAYER) | (class_ids == analyzer.cfg.GOALKEEPER)
        player_pixels = np.asarray(pixels[player_mask], dtype=np.float32)
        player_coords = np.asarray(result["pitch_coords"][player_mask], dtype=np.float32) if result.get("pitch_coords") is not None else np.zeros((0, 2), dtype=np.float32)

        opts = list(state.options)
        if state.valid and state.holder_pixel is not None and len(opts) > 0 and len(player_coords) > 0:
            holder_px = np.asarray(state.holder_pixel, dtype=np.float32)
            worst = min(opts, key=lambda p: float(p.get("optimal_score", 1.0)))
            for opt in opts[:5]:
                target_pitch = np.asarray(opt.get("target_pos"), dtype=np.float32)
                if target_pitch.shape != (2,):
                    continue
                d = np.linalg.norm(player_coords - target_pitch[None, :], axis=1)
                ridx = int(np.argmin(d))
                if float(d[ridx]) > 3.0:
                    continue
                target_px = player_pixels[ridx]
                color = _risk_color(opt, is_worst=bool(worst is opt))
                thick = 5 if bool(opt.get("is_best", False)) else 3
                p1 = tuple(map(int, holder_px))
                p2 = tuple(map(int, target_px))
                cv2.line(out, p1, p2, color, thick)
                cv2.circle(out, p2, 8 if bool(opt.get("is_best", False)) else 6, color, -1)
            cv2.circle(out, tuple(map(int, holder_px)), 20, (0, 255, 255), 3)

    cv2.putText(out, "Pass Risk:", (30, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.circle(out, (160, 28), 8, (0, 255, 0), -1)
    cv2.putText(out, "Best", (175, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.circle(out, (240, 28), 8, (0, 255, 255), -1)
    cv2.putText(out, "Med", (255, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.circle(out, (305, 28), 8, (0, 0, 255), -1)
    cv2.putText(out, "Risky", (320, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    kept = []
    for label in active_regret_labels:
        if frame_idx > int(label["end"]):
            continue
        dur = max(1, int(label["end"]) - int(label["start"]))
        rem = int(label["end"]) - frame_idx
        alpha = float(np.clip(rem / dur, 0.0, 1.0))
        x, y = label["pos"]
        y_txt = max(18, int(y + 26))
        color = (0, int(120 + 120 * alpha), int(255 * alpha))
        cv2.putText(out, label["text"], (int(x - 58), y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3)
        cv2.putText(out, label["text"], (int(x - 58), y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
        kept.append(label)
    active_regret_labels[:] = kept
    return out


def _draw_pitch_base(ax: Any) -> None:
    ax.set_facecolor("#2b7a3d")
    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(PITCH_WIDTH, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    white = (1, 1, 1, 0.95)
    ax.plot([0, PITCH_LENGTH], [0, 0], color=white, lw=2)
    ax.plot([0, PITCH_LENGTH], [PITCH_WIDTH, PITCH_WIDTH], color=white, lw=2)
    ax.plot([0, 0], [0, PITCH_WIDTH], color=white, lw=2)
    ax.plot([PITCH_LENGTH, PITCH_LENGTH], [0, PITCH_WIDTH], color=white, lw=2)
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH], color=white, lw=2)
    center = plt.Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15, fill=False, color=white, lw=2)
    ax.add_patch(center)
    pa_w = 40.32
    six_w = 18.32
    y0_pa = (PITCH_WIDTH - pa_w) / 2
    y0_six = (PITCH_WIDTH - six_w) / 2
    ax.add_patch(plt.Rectangle((0, y0_pa), 16.5, pa_w, fill=False, color=white, lw=2))
    ax.add_patch(plt.Rectangle((PITCH_LENGTH - 16.5, y0_pa), 16.5, pa_w, fill=False, color=white, lw=2))
    ax.add_patch(plt.Rectangle((0, y0_six), 5.5, six_w, fill=False, color=white, lw=2))
    ax.add_patch(plt.Rectangle((PITCH_LENGTH - 5.5, y0_six), 5.5, six_w, fill=False, color=white, lw=2))


def _save_heatmap_smooth(values: np.ndarray, out_path: Path, title: str, sigma: float = 7.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(values, dtype=np.float32)
    dense_w, dense_h = 320, 200
    dense = cv2.resize(data, (dense_w, dense_h), interpolation=cv2.INTER_CUBIC)
    smooth = gaussian_filter(dense, sigma=float(sigma))

    positive = smooth[smooth > 0]
    vmax = float(np.percentile(positive, 95)) if positive.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    heat = np.clip(smooth, 0.0, vmax)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=220)
    if Pitch is not None:
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            pitch_color="#2b7a3d",
            line_color="white",
            linewidth=2.2,
        )
        pitch.draw(ax=ax)
        ax.set_xlim(0, PITCH_LENGTH)
        ax.set_ylim(PITCH_WIDTH, 0)
    else:
        _draw_pitch_base(ax)

    im = ax.imshow(
        heat,
        extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0],
        cmap="RdYlGn_r",
        interpolation="bilinear",
        alpha=0.86,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )

    try:
        levels = np.linspace(0.0, vmax, 6)
        xx = np.linspace(0.0, PITCH_LENGTH, dense_w)
        yy = np.linspace(0.0, PITCH_WIDTH, dense_h)
        ax.contour(xx, yy, heat, levels=levels[1:], colors="white", linewidths=0.5, alpha=0.35)
    except Exception:
        pass

    ax.set_title(title, color="white", fontsize=17, fontweight="bold", pad=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Cumulative Regret", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _team_attack_left_to_right(team_id: int, timestamp_s: float) -> bool:
    # Team 1: left in 1H -> attacks right; right in 2H -> attacks left.
    # Team 0: right in 1H -> attacks left; left in 2H -> attacks right.
    second_half = bool(float(timestamp_s) >= SECOND_HALF_START_S)
    if int(team_id) == 1:
        return not second_half
    return second_half


def _canonicalize_event_xy(team_id: int, timestamp_s: float, x: float, y: float) -> Tuple[float, float]:
    x0 = float(np.clip(x, 0.0, PITCH_LENGTH))
    y0 = float(np.clip(y, 0.0, PITCH_WIDTH))
    attack_ltr = _team_attack_left_to_right(int(team_id), float(timestamp_s))
    if not attack_ltr:
        x0 = float(PITCH_LENGTH) - x0
    return x0, y0


def _half_bucket(timestamp_s: float) -> Optional[str]:
    t = float(timestamp_s)
    if t <= HALFTIME_START_S:
        return "first"
    if t >= SECOND_HALF_START_S:
        return "second"
    return None


def _save_heatmap_grid(values: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(values, dtype=np.float32)
    positive = data[data > 0]
    vmax = float(np.percentile(positive, 95)) if positive.size > 0 else 1.0
    vmax = max(vmax, 1e-6)
    heat = np.clip(data, 0.0, vmax)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    if Pitch is not None:
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=PITCH_LENGTH,
            pitch_width=PITCH_WIDTH,
            pitch_color="#2b7a3d",
            line_color="white",
            linewidth=2.0,
        )
        pitch.draw(ax=ax)
        ax.set_xlim(0, PITCH_LENGTH)
        ax.set_ylim(PITCH_WIDTH, 0)
    else:
        _draw_pitch_base(ax)

    im = ax.imshow(
        heat,
        extent=[0, PITCH_LENGTH, PITCH_WIDTH, 0],
        cmap="RdYlGn_r",
        interpolation="nearest",
        alpha=0.82,
        vmin=0.0,
        vmax=vmax,
        aspect="auto",
    )

    cell_w = PITCH_LENGTH / GRID_X
    cell_h = PITCH_WIDTH / GRID_Y
    for gx in range(1, GRID_X):
        x = gx * cell_w
        ax.plot([x, x], [0, PITCH_WIDTH], color=(1, 1, 1, 0.28), lw=0.8)
    for gy in range(1, GRID_Y):
        y = gy * cell_h
        ax.plot([0, PITCH_LENGTH], [y, y], color=(1, 1, 1, 0.28), lw=0.8)

    # One numeric value per cell.
    for iy in range(GRID_Y):
        for ix in range(GRID_X):
            v = float(data[iy, ix])
            x = (ix + 0.5) * cell_w
            y = (iy + 0.58) * cell_h
            txt = f"{v:.2f}"
            color = "white" if (vmax > 0 and v / vmax > 0.55) else "black"
            ax.text(x, y, txt, ha="center", va="center", fontsize=7, fontweight="bold", color=color)

    ax.set_title(title, color="white", fontsize=16, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Cumulative Regret", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.team is not None:
        print(f"[note] --team={args.team} provided, but both team heatmaps are generated automatically.")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ensure_model_weights(Path("data/models"))
    runtime = build_runtime()
    cfg = runtime["cfg"]
    SoccerAnalyzer = runtime["SoccerAnalyzer"]
    TEAM_COLORS = runtime["ctx"]["TEAM_COLORS"]
    BALL_COLOR = runtime["ctx"]["BALL_COLOR"]
    sv_mod = runtime["ctx"]["sv"]

    analyzer = SoccerAnalyzer(runtime["player_model"], runtime["pitch_model"], cfg)
    analyzer.ctx_sv = sv_mod

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    skip = max(1, int(args.skip_frames))
    out_fps = in_fps / skip

    video_out = out_dir / "match_analysis.mp4"
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open writer: {video_out}")

    regret_sum = {0: np.zeros((GRID_Y, GRID_X), dtype=np.float32), 1: np.zeros((GRID_Y, GRID_X), dtype=np.float32)}
    regret_half = {
        0: {"first": np.zeros((GRID_Y, GRID_X), dtype=np.float32), "second": np.zeros((GRID_Y, GRID_X), dtype=np.float32)},
        1: {"first": np.zeros((GRID_Y, GRID_X), dtype=np.float32), "second": np.zeros((GRID_Y, GRID_X), dtype=np.float32)},
    }
    pass_count = {0: np.zeros((GRID_Y, GRID_X), dtype=np.int32), 1: np.zeros((GRID_Y, GRID_X), dtype=np.int32)}
    pass_count_half = {
        0: {"first": np.zeros((GRID_Y, GRID_X), dtype=np.int32), "second": np.zeros((GRID_Y, GRID_X), dtype=np.int32)},
        1: {"first": np.zeros((GRID_Y, GRID_X), dtype=np.int32), "second": np.zeros((GRID_Y, GRID_X), dtype=np.int32)},
    }

    active_labels: List[Dict[str, Any]] = []
    prev_state: Optional[FrameState] = None
    last_event_frame = {0: -99999, 1: -99999}
    sampled_idx = 0
    raw_idx = 0
    pass_events = {0: 0, 1: 0}
    pass_events_half = {0: {"first": 0, "second": 0, "break": 0}, 1: {"first": 0, "second": 0, "break": 0}}
    skipped_no_ball: List[int] = []
    skipped_no_holder: List[int] = []
    skipped_goalkeeper_holder: List[int] = []
    ball_memory = BallMemory(
        last_pixel=None,
        last_pitch=None,
        vel_pixel=np.zeros((2,), dtype=np.float32),
        vel_pitch=np.zeros((2,), dtype=np.float32),
        miss_frames=0,
    )
    t0 = time.time()
    fade_frames = int(round(out_fps * 1.5))

    while True:
        if args.max_frames > 0 and raw_idx >= int(args.max_frames):
            break
        ok, frame = cap.read()
        if not ok:
            break

        if raw_idx % skip != 0:
            raw_idx += 1
            continue

        timestamp_s = float(raw_idx) / max(in_fps, 1e-6)
        result = analyzer.process_frame(frame)
        cur_state = _extract_state(
            result,
            analyzer,
            sampled_idx,
            timestamp_s,
            frame_shape=(height, width),
            ball_memory=ball_memory,
            runtime_ctx=runtime["ctx"],
        )

        if cur_state.skip_reason == "no_ball":
            skipped_no_ball.append(sampled_idx)
        elif cur_state.skip_reason == "no_holder":
            skipped_no_holder.append(sampled_idx)
        elif cur_state.skip_reason == "goalkeeper_holder":
            skipped_goalkeeper_holder.append(sampled_idx)

        for team_id in (0, 1):
            event = _detect_pass_regret_event(prev_state, cur_state, int(team_id), int(last_event_frame[team_id]))
            if event is None:
                continue
            px = event["from_pixel"]
            ex, ey = event["from_pitch"][0], event["from_pitch"][1]
            cx, cy = _canonicalize_event_xy(team_id, float(event["timestamp_s"]), float(ex), float(ey))
            ix, iy = _cell_index(cx, cy)
            regret_sum[team_id][iy, ix] += float(event["regret"])
            pass_count[team_id][iy, ix] += 1
            pass_events[team_id] += 1
            hb = _half_bucket(float(event["timestamp_s"]))
            if hb in ("first", "second"):
                regret_half[team_id][hb][iy, ix] += float(event["regret"])
                pass_count_half[team_id][hb][iy, ix] += 1
                pass_events_half[team_id][hb] += 1
            else:
                pass_events_half[team_id]["break"] += 1
            last_event_frame[team_id] = sampled_idx
            active_labels.append(
                {
                    "start": sampled_idx,
                    "end": sampled_idx + max(1, fade_frames),
                    "pos": (int(px[0]), int(px[1])),
                    "text": f"T{team_id} regret: {float(event['regret']):.2f}",
                }
            )

        annotated = _draw_match_frame(
            frame=frame,
            result=result,
            state=cur_state,
            analyzer=analyzer,
            team_colors=TEAM_COLORS,
            ball_color=BALL_COLOR,
            active_regret_labels=active_labels,
            frame_idx=sampled_idx,
        )
        writer.write(annotated)

        prev_state = cur_state
        sampled_idx += 1
        raw_idx += 1

        if sampled_idx % 500 == 0:
            elapsed = max(time.time() - t0, 1e-6)
            fps_eff = sampled_idx / elapsed
            if in_frames > 0:
                pct = 100.0 * min(raw_idx, in_frames) / in_frames
                print(
                    f"[progress] sampled={sampled_idx} raw={raw_idx}/{in_frames} ({pct:.1f}%) "
                    f"events_t0={pass_events[0]} events_t1={pass_events[1]} speed={fps_eff:.2f} fps"
                )
            else:
                print(
                    f"[progress] sampled={sampled_idx} raw={raw_idx} "
                    f"events_t0={pass_events[0]} events_t1={pass_events[1]} speed={fps_eff:.2f} fps"
                )

    cap.release()
    writer.release()

    match_title = video_path.stem.replace("_", " ")
    _save_heatmap_smooth(
        regret_sum[0],
        out_dir / "regret_heatmap_team0.png",
        title=f"{match_title} | Team 0 (Red) Regret Heatmap",
    )
    _save_heatmap_smooth(
        regret_sum[1],
        out_dir / "regret_heatmap_team1.png",
        title=f"{match_title} | Team 1 (Blue) Regret Heatmap",
    )
    # Per-half outputs (smooth + grid)
    for team_id in (0, 1):
        for half_key, half_label in (("first", "First Half"), ("second", "Second Half")):
            _save_heatmap_grid(
                regret_half[team_id][half_key],
                out_dir / f"regret_team{team_id}_{half_key}_half_grid.png",
                title=f"{match_title} | Team {team_id} Regret Grid ({half_label})",
            )

    analyzed_minutes = (sampled_idx / max(out_fps, 1e-6)) / 60.0
    total_events = int(pass_events[0] + pass_events[1])
    rate = float(total_events / max(analyzed_minutes, 1e-6))
    print(
        f"[stats] total_frames_processed={sampled_idx} pass_events_total={total_events} "
        f"pass_events_t0={pass_events[0]} pass_events_t1={pass_events[1]} passes_per_min={rate:.2f}"
    )
    print(
        "[stats] team0_events_by_half: "
        f"first={pass_events_half[0]['first']} second={pass_events_half[0]['second']} break={pass_events_half[0]['break']}"
    )
    print(
        "[stats] team1_events_by_half: "
        f"first={pass_events_half[1]['first']} second={pass_events_half[1]['second']} break={pass_events_half[1]['break']}"
    )
    if rate < 3.0:
        print("[warning] pass detection rate below 3 passes/minute; investigate no-ball/no-holder gaps.")

    def _summarize_frames(name: str, frames: List[int]) -> None:
        if len(frames) == 0:
            print(f"[stats] {name}: 0")
            return
        preview = ", ".join(str(v) for v in frames[:40])
        suffix = " ..." if len(frames) > 40 else ""
        print(f"[stats] {name}: {len(frames)} | first_frames: {preview}{suffix}")

    _summarize_frames("skipped_no_ball", skipped_no_ball)
    _summarize_frames("skipped_no_holder", skipped_no_holder)
    _summarize_frames("skipped_goalkeeper_holder", skipped_goalkeeper_holder)

    print(f"[done] sampled_frames={sampled_idx}")
    print(f"[done] outputs: {out_dir / 'regret_heatmap_team0.png'}")
    print(f"[done] outputs: {out_dir / 'regret_heatmap_team1.png'}")
    print(f"[done] outputs: {out_dir / 'regret_team0_first_half_grid.png'}")
    print(f"[done] outputs: {out_dir / 'regret_team0_second_half_grid.png'}")
    print(f"[done] outputs: {out_dir / 'regret_team1_first_half_grid.png'}")
    print(f"[done] outputs: {out_dir / 'regret_team1_second_half_grid.png'}")
    print(f"[done] outputs: {video_out}")


if __name__ == "__main__":
    main()
