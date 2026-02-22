#!/usr/bin/env python3
"""Runtime patches: stable team tags + strict ball-carrier pass rendering + team-relative xT scoring."""

from __future__ import annotations

from typing import Any, Dict, Optional


def apply_runtime_overrides(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Patch notebook runtime in-place without editing notebook cells."""
    if "SoccerAnalyzer" not in ctx or "classify_teams" not in ctx:
        return ctx

    np = ctx["np"]
    cv2 = ctx["cv2"]
    sv = ctx["sv"]
    SoccerAnalyzer = ctx["SoccerAnalyzer"]
    TEAM_COLORS = ctx["TEAM_COLORS"]
    BALL_COLOR = ctx["BALL_COLOR"]
    XT_GRID = np.asarray(ctx["XT_GRID"], dtype=np.float32)
    get_xt = ctx["get_xt"]

    def _to_track_id(raw_id) -> Optional[int]:
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

    def _weighted_dist(a: Any, b: Any) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        if a.shape != b.shape:
            return 9.9
        w = np.asarray([2.3, 1.4, 1.4, 0.9, 0.7, 1.2, 1.0, 1.0, 0.5, 0.5], dtype=np.float32)
        if w.shape[0] != a.shape[0]:
            w = np.ones_like(a, dtype=np.float32)
        return float(np.linalg.norm((a - b) * w))

    def _feature_hue_hsv(f: Any) -> tuple[float, float, float, float]:
        arr = np.asarray(f, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 8:
            return 0.0, 0.0, 0.0, 0.0
        l = float(arr[0])   # normalized lightness-ish
        s = float(arr[3])   # normalized saturation
        v = float(arr[4])   # normalized value
        c = float(arr[6])
        si = float(arr[7])
        hue_rad = float(np.arctan2(si, c))
        if hue_rad < 0:
            hue_rad += float(2.0 * np.pi)
        hue_cv = float(hue_rad * (180.0 / (2.0 * np.pi)))  # OpenCV hue range proxy [0,180)
        return hue_cv, s, v, l

    def _hue_circ_dist(h: float, target: float) -> float:
        d = abs(float(h) - float(target))
        d = min(d, 180.0 - d)
        return float(d / 90.0)  # normalized ~[0,1]

    def _color_cost_from_feature(f: Any, color_name: str) -> float:
        h, s, v, l = _feature_hue_hsv(f)
        color = str(color_name).lower()
        if color == "white":
            # White shirts: low saturation + high luminance/value.
            return float(2.7 * max(0.0, s - 0.12) + 1.4 * max(0.0, 0.72 - v) + 0.8 * max(0.0, 0.66 - l))
        if color == "red":
            hd = min(_hue_circ_dist(h, 0.0), _hue_circ_dist(h, 179.0))
            return float(2.3 * hd + 1.1 * max(0.0, 0.26 - s) + 0.5 * max(0.0, 0.35 - v))
        if color == "yellow":
            hd = _hue_circ_dist(h, 26.0)
            return float(2.0 * hd + 0.9 * max(0.0, 0.28 - s) + 0.6 * max(0.0, 0.50 - v))
        if color == "blue":
            hd = _hue_circ_dist(h, 108.0)
            return float(2.2 * hd + 0.8 * max(0.0, 0.24 - s) + 0.4 * max(0.0, 0.25 - v))
        return 9.9

    def _scenario_color_pair(self: Any) -> Optional[tuple[str, str]]:
        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        mapping = {
            "scenario2": ("red", "white"),
            "scenario3": ("red", "yellow"),
            "scenario4": ("red", "skyblue"),
        }
        return mapping.get(scenario_name)

    def _scenario_color_predict(self: Any, f: Any) -> Optional[tuple[int, float]]:
        pair = _scenario_color_pair(self)
        if pair is None:
            return None
        c0 = _color_cost_from_feature(f, pair[0])
        c1 = _color_cost_from_feature(f, pair[1])
        margin = abs(c0 - c1)
        pred = 0 if c0 <= c1 else 1
        return int(pred), float(margin)

    def _extract_kit_feature(frame: Any, xyxy: Any) -> Optional[Any]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if (x2 - x1) < 10 or (y2 - y1) < 16:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]

        yy1 = int(0.16 * ch)
        yy2 = int(0.64 * ch)
        xx1 = int(0.18 * cw)
        xx2 = int(0.82 * cw)
        jersey = crop[yy1:yy2, xx1:xx2]
        if jersey.size == 0:
            return None

        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
        hue = hsv[..., 0].astype(np.float32)
        sat = hsv[..., 1].astype(np.float32)
        val = hsv[..., 2].astype(np.float32)
        labf = lab.astype(np.float32)

        green = (hue >= 28.0) & (hue <= 98.0) & (sat >= 25.0)
        good = (sat > 18.0) & (val > 32.0) & (~green)
        if int(np.sum(good)) < 24:
            good = val > 38.0
        if int(np.sum(good)) < 16:
            return None

        h_vals = hue[good]
        s_vals = sat[good]
        v_vals = val[good]
        l_vals = labf[..., 0][good]
        a_vals = labf[..., 1][good]
        b_vals = labf[..., 2][good]
        hue_rad = (h_vals / 180.0) * (2.0 * np.pi)
        sat_w = np.clip(s_vals / 255.0, 0.05, 1.0)
        l_norm = l_vals / 255.0
        chroma = np.sqrt((a_vals / 255.0 - 0.5) ** 2 + (b_vals / 255.0 - 0.5) ** 2)

        return np.array(
            [
                float(np.median(l_norm)),
                float(np.median(a_vals) / 255.0),
                float(np.median(b_vals) / 255.0),
                float(np.median(s_vals) / 255.0),
                float(np.median(v_vals) / 255.0),
                float(np.median(chroma)),
                float(np.median(np.cos(hue_rad) * sat_w)),
                float(np.median(np.sin(hue_rad) * sat_w)),
                float(np.percentile(l_norm, 90)),
                float(np.percentile(l_norm, 10)),
            ],
            dtype=np.float32,
        )

    def _extract_jersey_crop(frame: Any, xyxy: Any) -> Optional[Any]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if (x2 - x1) < 10 or (y2 - y1) < 16:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        # Tighter torso-core crop to avoid shorts/socks and reduce sleeve contamination.
        yy1 = int(0.14 * ch)
        yy2 = int(0.42 * ch)
        xx1 = int(0.28 * cw)
        xx2 = int(0.72 * cw)
        jersey = crop[yy1:yy2, xx1:xx2]
        if jersey.size == 0 or jersey.shape[0] < 6 or jersey.shape[1] < 6:
            return None
        return jersey

    def _scenario_color_scores_from_crop(self: Any, frame: Any, xyxy: Any) -> Optional[tuple[float, float]]:
        pair = _scenario_color_pair(self)
        if pair is None:
            return None
        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        jersey = _extract_jersey_crop(frame, xyxy)
        if jersey is None:
            return None

        hh, ww = jersey.shape[:2]
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
        hue = hsv[..., 0].astype(np.float32)
        sat = hsv[..., 1].astype(np.float32)
        val = hsv[..., 2].astype(np.float32)
        L = lab[..., 0].astype(np.float32)
        A = lab[..., 1].astype(np.float32)
        B = lab[..., 2].astype(np.float32)

        hh, ww = hue.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        cx = (ww - 1) * 0.5
        cy = (hh - 1) * 0.48
        wx = max(2.5, ww * 0.22)
        wy = max(2.5, hh * 0.26)
        center_w = np.exp(-0.5 * (((xx - cx) / wx) ** 2 + ((yy - cy) / wy) ** 2)).astype(np.float32)

        green = (hue >= 28.0) & (hue <= 98.0) & (sat >= 30.0)
        valid = (~green) & (val >= 35.0)
        if int(np.sum(valid)) < 20:
            return None
        weights = center_w[valid]
        if weights.size == 0:
            return None
        wsum = float(np.sum(weights)) + 1e-6
        hue = hue[valid]
        sat = sat[valid]
        val = val[valid]
        L = L[valid]
        A = A[valid]
        B = B[valid]
        wv = weights.astype(np.float32)

        def _wmean(x: Any) -> float:
            xa = np.asarray(x, dtype=np.float32)
            return float(np.sum(xa * wv) / wsum)

        def _wfrac(mask: Any) -> float:
            ma = np.asarray(mask, dtype=np.float32)
            return float(np.sum(ma * wv) / wsum)

        def _score(name: str) -> float:
            name = str(name).lower()
            if name == "red":
                m = (((hue <= 12.0) | (hue >= 170.0)) & (sat >= 65.0) & (val >= 45.0))
                ratio = _wfrac(m)
                red_lab = _wmean(np.clip((A - 128.0) / 48.0, 0.0, 1.0))
                deep_red = (((hue <= 8.0) | (hue >= 174.0)) & (sat >= 90.0) & (val >= 55.0))
                base = 0.66 * ratio + 0.18 * _wfrac(deep_red) + 0.16 * red_lab
                # Scenario3 is Arsenal (red/white shirts) vs yellow. Reward red+white torso mixture.
                if scenario_name == "scenario3":
                    white_mix = _wfrac((sat <= 88.0) & (val >= 100.0) & (L >= 105.0))
                    red_like = _wfrac((((hue <= 12.0) | (hue >= 170.0)) & (sat >= 48.0) & (val >= 38.0)))
                    # red stripe + bright white panel is a strong Arsenal signal in this clip.
                    striped = min(float(max(ratio, red_like)), float(white_mix))
                    base += 0.35 * striped + 0.16 * white_mix
                    if red_like >= 0.055 and white_mix >= 0.11:
                        base = max(base, 0.78)
                return float(base)
            if name == "white":
                # White kits in old broadcasts often have dark/navy trim; accept "mostly bright low-sat" pixels.
                m = (sat <= 78.0) & (val >= 95.0) & (L >= 110.0)
                ratio = _wfrac(m)
                low_sat = _wmean(np.clip((105.0 - sat) / 105.0, 0.0, 1.0))
                bright = _wmean(np.clip((val - 75.0) / 180.0, 0.0, 1.0))
                lab_bright = _wmean(np.clip((L - 95.0) / 160.0, 0.0, 1.0))
                neutral_ab = _wmean(np.clip(1.0 - (np.abs(A - 128.0) + np.abs(B - 128.0)) / 70.0, 0.0, 1.0))
                return 0.42 * ratio + 0.18 * low_sat + 0.18 * bright + 0.12 * lab_bright + 0.10 * neutral_ab
            if name == "yellow":
                m = (hue >= 14.0) & (hue <= 38.0) & (sat >= 85.0) & (val >= 95.0)
                ratio = _wfrac(m)
                y = 0.72 * ratio + 0.14 * _wmean(np.clip((sat - 70.0) / 140.0, 0.0, 1.0)) + 0.14 * _wmean(np.clip((val - 80.0) / 175.0, 0.0, 1.0))
                if scenario_name == "scenario3":
                    # Penalize red contamination so red/white striped shirts don't get labeled yellow.
                    red_like = _wfrac((((hue <= 12.0) | (hue >= 170.0)) & (sat >= 55.0) & (val >= 45.0)))
                    white_mix = _wfrac((sat <= 88.0) & (val >= 100.0) & (L >= 105.0))
                    striped = min(float(red_like), float(white_mix))
                    y -= 0.55 * red_like + 0.30 * striped
                return float(max(y, 0.0))
            if name == "skyblue":
                # Light blue / cyan kits (e.g. Manchester City) are bright with moderate saturation.
                m_core = (hue >= 72.0) & (hue <= 118.0) & (sat >= 12.0) & (val >= 125.0)
                m_soft = (hue >= 64.0) & (hue <= 126.0) & (sat >= 18.0) & (val >= 95.0)
                ratio_core = _wfrac(m_core)
                ratio_soft = _wfrac(m_soft)
                cool_b = _wmean(np.clip((128.0 - B) / 58.0, 0.0, 1.0))
                cool_a = _wmean(np.clip((128.0 - A) / 44.0, 0.0, 1.0))
                bright = _wmean(np.clip((val - 95.0) / 160.0, 0.0, 1.0))
                sat_soft = _wmean(np.clip((sat - 8.0) / 110.0, 0.0, 1.0))
                return 0.34 * ratio_core + 0.22 * ratio_soft + 0.17 * cool_b + 0.11 * cool_a + 0.10 * bright + 0.06 * sat_soft
            if name == "blue":
                # Accept both deep blue and sky/cyan kits (e.g., Manchester City light blue).
                m_deep = (hue >= 92.0) & (hue <= 132.0) & (sat >= 45.0) & (val >= 40.0)
                m_sky = (hue >= 76.0) & (hue <= 118.0) & (sat >= 18.0) & (val >= 120.0)
                r_deep = _wfrac(m_deep)
                r_sky = _wfrac(m_sky)
                blue_lab = _wmean(np.clip((128.0 - B) / 52.0, 0.0, 1.0))
                cyan_lab = _wmean(np.clip((128.0 - A) / 40.0, 0.0, 1.0)) * _wmean(
                    np.clip((128.0 - B) / 56.0, 0.0, 1.0)
                )
                sat_term = _wmean(np.clip((sat - 15.0) / 120.0, 0.0, 1.0))
                val_term = _wmean(np.clip((val - 80.0) / 170.0, 0.0, 1.0))
                return 0.34 * r_deep + 0.30 * r_sky + 0.16 * blue_lab + 0.10 * cyan_lab + 0.05 * sat_term + 0.05 * val_term
            return 0.0

        s0 = float(_score(pair[0]))
        s1 = float(_score(pair[1]))
        if max(s0, s1) <= 0.05:
            return None
        return float(s0), float(s1)

    def _scenario_color_predict_from_crop(self: Any, frame: Any, xyxy: Any) -> Optional[tuple[int, float]]:
        scores = _scenario_color_scores_from_crop(self, frame, xyxy)
        if scores is None:
            return None
        s0, s1 = scores
        pred = 0 if s0 >= s1 else 1
        margin = abs(s0 - s1)
        return int(pred), float(margin)

    def _scenario_forced_label_from_crop(self: Any, frame: Any, xyxy: Any) -> Optional[tuple[int, float]]:
        """Scenario-specific hard overrides for jersey colors when the normal scorer is still too soft."""
        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        if scenario_name != "scenario3":
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if (x2 - x1) < 10 or (y2 - y1) < 16:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        # Wider torso window than the generic scorer so Arsenal's red torso + white sleeves
        # remains visible even under blur/rotation.
        yy1 = int(0.10 * ch)
        yy2 = int(0.58 * ch)
        xx1 = int(0.12 * cw)
        xx2 = int(0.88 * cw)
        jersey = crop[yy1:yy2, xx1:xx2]
        if jersey.size == 0 or jersey.shape[0] < 6 or jersey.shape[1] < 6:
            return None
        hh, ww = jersey.shape[:2]
        hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
        hue2 = hsv[..., 0].astype(np.float32)
        sat2 = hsv[..., 1].astype(np.float32)
        val2 = hsv[..., 2].astype(np.float32)
        L2 = lab[..., 0].astype(np.float32)
        # Basic mask, ignore grass/very dark pixels.
        valid2 = (val2 >= 35.0)
        if ww >= 9:
            left = slice(0, max(1, ww // 3))
            center = slice(max(0, ww // 3), min(ww, (2 * ww) // 3))
            right = slice(min(ww - 1, (2 * ww) // 3), ww)
            red2 = (((hue2 <= 14.0) | (hue2 >= 168.0)) & (sat2 >= 34.0) & (val2 >= 34.0) & valid2)
            white2 = ((sat2 <= 96.0) & (val2 >= 95.0) & (L2 >= 100.0) & valid2)
            red_edges = float(
                0.5 * np.mean(red2[:, left].astype(np.float32))
                + 0.5 * np.mean(red2[:, right].astype(np.float32))
            )
            white_center = float(np.mean(white2[:, center].astype(np.float32)))
        else:
            red_edges = 0.0
            white_center = 0.0

        valid = valid2
        if int(np.sum(valid)) < 20:
            return None
        hue = hue2[valid]
        sat = sat2[valid]
        val = val2[valid]
        L = L2[valid]
        red_mask = (((hue <= 14.0) | (hue >= 168.0)) & (sat >= 34.0) & (val >= 34.0))
        red_like = float(np.mean(red_mask.astype(np.float32)))
        red_strong = float(np.mean((((hue <= 10.0) | (hue >= 172.0)) & (sat >= 70.0) & (val >= 48.0)).astype(np.float32)))
        white_mask = ((sat <= 96.0) & (val >= 95.0) & (L >= 100.0))
        white_like = float(np.mean(white_mask.astype(np.float32)))
        yellow_mask = ((hue >= 14.0) & (hue <= 40.0) & (sat >= 70.0) & (val >= 85.0))
        yellow_like = float(np.mean(yellow_mask.astype(np.float32)))

        # Side-vs-center structure check for Arsenal red body + white sleeves.
        if ww < 9:
            red_edges = red_like
            white_center = white_like

        # Arsenal striped shirt: visible red stripes + bright white panel/sleeves.
        striped_red = max(
            min(max(red_like, red_strong), white_like),
            min(max(red_edges, red_like), max(white_center, 0.75 * white_like)),
        )
        if (
            striped_red >= 0.050
            or (red_like >= 0.028 and white_like >= 0.14)
            or (red_edges >= 0.030 and white_center >= 0.12)
        ):
            # team 0 for scenario3 mapping ("red", "yellow")
            conf = 0.45 + 0.9 * float(striped_red)
            return 0, float(conf)

        # Strong yellow, with almost no red contamination => team 1
        if yellow_like >= 0.22 and red_like <= 0.015 and white_like <= 0.22:
            conf = 0.35 + 0.8 * float(yellow_like)
            return 1, float(conf)
        return None

    def _scenario_track_color_label(
        self: Any,
        tid: Optional[int],
        pred: int,
        s0: float,
        s1: float,
        frame_idx: int,
    ) -> int:
        if tid is None or int(pred) not in (0, 1):
            return int(pred)
        stats = getattr(self, "_rt_track_color_stats", None)
        if stats is None:
            stats = {}
            self._rt_track_color_stats = stats
        row = stats.get(int(tid))
        if row is None:
            row = {
                "sum": np.zeros((2,), dtype=np.float32),
                "n": 0,
                "lock": -1,
                "last": int(frame_idx),
            }
        # Reset if track disappeared for a while (tracker may recycle IDs).
        if (int(frame_idx) - int(row.get("last", frame_idx))) > 25:
            row["sum"] = np.zeros((2,), dtype=np.float32)
            row["n"] = 0
            row["lock"] = -1
        row["sum"] = 0.94 * np.asarray(row["sum"], dtype=np.float32) + np.asarray([float(s0), float(s1)], dtype=np.float32)
        row["n"] = int(row.get("n", 0)) + 1
        row["last"] = int(frame_idx)

        # Hard lock once cumulative evidence is clear. This is intentionally strict for short scenario clips.
        if int(row.get("lock", -1)) not in (0, 1):
            ss = np.asarray(row["sum"], dtype=np.float32)
            total = float(np.sum(ss))
            margin = float(abs(ss[0] - ss[1]) / max(total, 1e-6))
            best = int(np.argmax(ss))
            # Delay hard-lock to avoid early wrong lock in cluttered frames.
            if row["n"] >= 6 and margin >= 0.32 and float(ss[best]) >= 1.10:
                row["lock"] = best

        stats[int(tid)] = row
        if int(row.get("lock", -1)) in (0, 1):
            return int(row["lock"])

        ss = np.asarray(row["sum"], dtype=np.float32)
        return int(np.argmax(ss)) if float(np.sum(ss)) > 0.0 else int(pred)

    def _stable_track_team(self: Any, tid: Optional[int]) -> int:
        if tid is None:
            return -1
        votes = getattr(self, "_rt_track_votes", {}).get(tid)
        if votes is None:
            return -1
        total = float(votes[0] + votes[1])
        if total < 5.0:
            return -1
        best = int(np.argmax(votes))
        ratio = float(votes[best] / max(total, 1e-6))
        return best if ratio >= 0.74 else -1

    def _update_track_votes(self: Any, tid: Optional[int], team: int, frame_idx: int) -> None:
        if tid is None or int(team) not in (0, 1):
            return
        votes = getattr(self, "_rt_track_votes", None)
        seen = getattr(self, "_rt_track_seen", None)
        if votes is None or seen is None:
            votes = {}
            seen = {}
            self._rt_track_votes = votes
            self._rt_track_seen = seen
        row = votes.get(tid)
        if row is None:
            row = np.zeros((2,), dtype=np.float32)
        row *= 0.92
        row[int(team)] += 1.0
        votes[tid] = row
        seen[tid] = int(frame_idx)

    def _prune_track_votes(self: Any, frame_idx: int) -> None:
        seen = getattr(self, "_rt_track_seen", None)
        votes = getattr(self, "_rt_track_votes", None)
        if seen is None or votes is None:
            return
        stale = [tid for tid, last in seen.items() if (int(frame_idx) - int(last)) > 220]
        for tid in stale:
            seen.pop(tid, None)
            votes.pop(tid, None)
        # Keep hysteresis state aligned with active tracks.
        hys = getattr(self, "_rt_team_hysteresis", None)
        if hys is not None:
            for tid in stale:
                hys.pop(tid, None)
        color_stats = getattr(self, "_rt_track_color_stats", None)
        if color_stats is not None:
            for tid in stale:
                color_stats.pop(tid, None)

    def _hysteresis_team_assign(
        self: Any,
        tid: Optional[int],
        pred_team: int,
        frame_idx: int,
        change_window: int = 8,
        margin: float = 1.0,
        min_switch_margin: float = 0.16,
    ) -> int:
        """Per-track label hysteresis to suppress frame-to-frame team color flips."""
        if tid is None or int(pred_team) not in (0, 1):
            return int(pred_team)

        state_map = getattr(self, "_rt_team_hysteresis", None)
        if state_map is None:
            state_map = {}
            self._rt_team_hysteresis = state_map

        s = state_map.get(int(tid))
        if s is None:
            s = {"label": int(pred_team), "cand": -1, "count": 0, "last_frame": int(frame_idx)}
            state_map[int(tid)] = s
            return int(pred_team)

        # Reset pending candidate if the track disappeared for a while.
        if (int(frame_idx) - int(s.get("last_frame", frame_idx))) > 3:
            s["cand"] = -1
            s["count"] = 0

        current = int(s.get("label", pred_team))
        pred = int(pred_team)

        if pred == current:
            s["cand"] = -1
            s["count"] = 0
        else:
            # Do not even start a team switch unless the color evidence is clearly separated.
            if float(margin) < float(min_switch_margin):
                s["cand"] = -1
                s["count"] = 0
                s["last_frame"] = int(frame_idx)
                return current

            if int(s.get("cand", -1)) == pred:
                s["count"] = int(s.get("count", 0)) + 1
            else:
                s["cand"] = pred
                s["count"] = 1

            if int(s["count"]) >= max(1, int(change_window)):
                s["label"] = pred
                s["cand"] = -1
                s["count"] = 0

        s["last_frame"] = int(frame_idx)
        return int(s.get("label", pred))

    def _relabel_teams(self: Any, frame: Any, result: Dict[str, Any]) -> None:
        det = result.get("detections")
        teams_in = result.get("teams")
        if det is None or teams_in is None or len(det) == 0:
            return

        teams = np.asarray(teams_in, dtype=int).copy()
        class_ids = np.asarray(det.class_id)
        player_mask = (class_ids == self.cfg.PLAYER) | (class_ids == self.cfg.GOALKEEPER)
        player_idx = np.where(player_mask)[0]
        if len(player_idx) < 4:
            result["teams"] = teams
            return

        frame_idx = int(getattr(self, "_rt_frame_idx", 0))
        tracker_ids = getattr(det, "tracker_id", None)
        feats = {}
        feat_idx = []
        features = []

        for i in player_idx:
            f = _extract_kit_feature(frame, det.xyxy[int(i)])
            if f is None:
                continue
            feats[int(i)] = f
            feat_idx.append(int(i))
            features.append(f)

        if len(feat_idx) < 4:
            if tracker_ids is not None:
                for i in player_idx:
                    tid = _to_track_id(tracker_ids[int(i)])
                    stable = _stable_track_team(self, tid)
                    if stable in (0, 1):
                        teams[int(i)] = stable
            result["teams"] = teams
            return

        data = np.asarray(features, dtype=np.float32)

        # Scenario-specific jersey-color classifier for known hard clips.
        scenario_pair = _scenario_color_pair(self)
        if scenario_pair is not None:
            assigned_direct = 0
            for det_i in feat_idx:
                forced_pack = _scenario_forced_label_from_crop(self, frame, det.xyxy[int(det_i)])
                crop_scores = _scenario_color_scores_from_crop(self, frame, det.xyxy[int(det_i)])
                pred_pack = None
                if forced_pack is not None:
                    pred_pack = forced_pack
                if crop_scores is not None:
                    s0c, s1c = crop_scores
                    crop_pred_pack = (0 if s0c >= s1c else 1, abs(float(s0c) - float(s1c)))
                    if pred_pack is None:
                        pred_pack = crop_pred_pack
                    else:
                        # If forced and crop agree, strengthen confidence; if they disagree keep forced.
                        if int(pred_pack[0]) == int(crop_pred_pack[0]):
                            pred_pack = (int(pred_pack[0]), float(max(pred_pack[1], crop_pred_pack[1] + 0.12)))
                if pred_pack is None and str(getattr(self, "_rt_scenario_name", "")).lower() != "scenario4":
                    f = feats.get(int(det_i))
                    if f is not None:
                        pred_pack = _scenario_color_predict(self, f)
                old = int(teams[int(det_i)])
                tid = _to_track_id(tracker_ids[int(det_i)]) if tracker_ids is not None else None
                stable = _stable_track_team(self, tid)
                if pred_pack is None:
                    # Strict scenario mode: do not force a label from kmeans later.
                    # Prefer stable historical label, otherwise keep existing/unknown.
                    if stable in (0, 1):
                        teams[int(det_i)] = stable
                    elif old in (0, 1):
                        teams[int(det_i)] = old
                    continue
                pred, color_margin = pred_pack
                if str(getattr(self, "_rt_scenario_name", "")).lower() == "scenario3" and crop_scores is not None:
                    s0c, s1c = float(crop_scores[0]), float(crop_scores[1])
                    # For Arsenal (red/white) vs yellow, be very strict before assigning yellow.
                    # Red/white stripes can look yellow under blur/compression if we allow weak yellow wins.
                    if int(pred) == 1:
                        redish_enough = s0c >= 0.26
                        weak_yellow = s1c < 0.44 or (s1c - s0c) < 0.18
                        if redish_enough or weak_yellow:
                            if stable in (0, 1):
                                pred = int(stable)
                            elif old in (0, 1):
                                pred = int(old)
                            else:
                                continue
                            color_margin = max(color_margin, 0.26)
                    else:
                        # When red wins, keep it sticky even on modest margins if there is visible red evidence.
                        if s0c >= 0.24 and (s0c + 0.02) >= s1c:
                            color_margin = max(color_margin, 0.28)
                # Very strict: only keep weak predictions if supported by old/stable labels.
                if color_margin < 0.14:
                    if old in (0, 1):
                        pred = old
                    else:
                        if stable in (0, 1):
                            pred = stable
                        else:
                            continue
                elif crop_scores is not None and tid is not None:
                    pred = _scenario_track_color_label(self, tid, int(pred), float(crop_scores[0]), float(crop_scores[1]), frame_idx)
                elif stable in (0, 1) and color_margin < 0.20:
                    pred = stable
                # Scenario3 (Arsenal red/white striped vs yellow): once stable, avoid switching on weak/noisy evidence.
                if str(getattr(self, "_rt_scenario_name", "")).lower() == "scenario3" and stable in (0, 1):
                    if int(pred) != int(stable) and float(color_margin) < 0.34:
                        pred = stable
                if tid is not None:
                    pred = _hysteresis_team_assign(
                        self,
                        tid,
                        int(pred),
                        frame_idx,
                        change_window=8,
                        margin=float(max(color_margin, 0.24)),
                        min_switch_margin=0.20,
                    )
                teams[int(det_i)] = int(pred)
                assigned_direct += 1

            # Fill remaining from stable tracks if available.
            if tracker_ids is not None:
                for i in player_idx:
                    if int(teams[int(i)]) in (0, 1):
                        continue
                    tid = _to_track_id(tracker_ids[int(i)])
                    stable = _stable_track_team(self, tid)
                    if stable in (0, 1):
                        teams[int(i)] = stable
            # For these manually-specified jersey-color scenarios, never fallback to kmeans.
            # Kmeans causes cross-team flips under crowd/background shifts in old broadcasts.
            # Ambiguous players remain unchanged or grey until evidence stabilizes.

        if tracker_ids is not None and scenario_pair is None:
            for i in player_idx:
                tid = _to_track_id(tracker_ids[int(i)])
                stable = _stable_track_team(self, tid)
                if stable in (0, 1):
                    teams[int(i)] = stable

        if scenario_pair is None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 28, 0.12)
            _, labels, centers = cv2.kmeans(data, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape(-1)

            cluster_votes = {0: np.zeros((2,), dtype=np.float32), 1: np.zeros((2,), dtype=np.float32)}
            for local_i, det_i in enumerate(feat_idx):
                t = int(teams[det_i])
                if t in (0, 1):
                    cluster_votes[int(labels[local_i])][t] += 1.0

            mapping: Dict[int, int] = {}
            for c in (0, 1):
                if float(np.sum(cluster_votes[c])) > 0:
                    mapping[c] = int(np.argmax(cluster_votes[c]))

            proto = getattr(self, "_rt_team_proto", None)
            if proto is not None and len(mapping) < 2:
                for c in (0, 1):
                    if c in mapping:
                        continue
                    d0 = _weighted_dist(centers[c], proto[0])
                    d1 = _weighted_dist(centers[c], proto[1])
                    mapping[c] = 0 if d0 <= d1 else 1

            if len(mapping) < 2:
                order = np.argsort(centers[:, 0])
                mapping.setdefault(int(order[0]), 1)
                mapping.setdefault(int(order[1]), 0)
            if mapping.get(0, -1) == mapping.get(1, -1):
                mapping[1] = 1 - int(mapping.get(0, 0))

            team_centers = np.zeros_like(centers, dtype=np.float32)
            for c in (0, 1):
                team_centers[int(mapping[c])] = centers[c]

            for local_i, det_i in enumerate(feat_idx):
                c = int(labels[local_i])
                pred = int(mapping[c])
                f = feats[int(det_i)]
                d0 = _weighted_dist(f, team_centers[0])
                d1 = _weighted_dist(f, team_centers[1])
                margin = abs(d0 - d1) / max(d0 + d1, 1e-6)

                old = int(teams[int(det_i)])
                tid = _to_track_id(tracker_ids[int(det_i)]) if tracker_ids is not None else None
                stable = _stable_track_team(self, tid)

                if stable in (0, 1) and stable != pred and margin < 0.18:
                    pred = stable
                elif old in (0, 1) and old != pred and margin < 0.10:
                    pred = old
                # Once a track has a stable historical label, strongly prefer it.
                # This prevents late-clip flicker in dense clusters.
                if stable in (0, 1):
                    pred = stable
                # Stronger anti-jitter hysteresis: only switch team after consecutive confirmations.
                if tid is not None and pred in (0, 1):
                    pred = _hysteresis_team_assign(
                        self,
                        tid,
                        pred,
                        frame_idx,
                        change_window=8,
                        margin=float(margin),
                        min_switch_margin=0.16,
                    )
                teams[int(det_i)] = pred

        if tracker_ids is not None:
            for i in player_idx:
                if int(teams[int(i)]) in (0, 1):
                    continue
                tid = _to_track_id(tracker_ids[int(i)])
                stable = _stable_track_team(self, tid)
                if stable in (0, 1):
                    teams[int(i)] = stable

        known = teams[player_idx]
        known = known[known != -1]
        if len(known) >= 10:
            c0 = int(np.sum(known == 0))
            c1 = int(np.sum(known == 1))
            if min(c0, c1) == 0 and tracker_ids is not None:
                for i in player_idx:
                    tid = _to_track_id(tracker_ids[int(i)])
                    stable = _stable_track_team(self, tid)
                    if stable in (0, 1):
                        teams[int(i)] = stable

        team_feats = {0: [], 1: []}
        for det_i in feat_idx:
            t = int(teams[int(det_i)])
            if t in (0, 1):
                team_feats[t].append(feats[int(det_i)])

        if len(team_feats[0]) >= 2 and len(team_feats[1]) >= 2:
            f0 = np.median(np.asarray(team_feats[0], dtype=np.float32), axis=0)
            f1 = np.median(np.asarray(team_feats[1], dtype=np.float32), axis=0)
            prev = getattr(self, "_rt_team_proto", None)
            if prev is None:
                self._rt_team_proto = np.asarray([f0, f1], dtype=np.float32)
            else:
                prev = np.asarray(prev, dtype=np.float32)
                self._rt_team_proto = (0.82 * prev + 0.18 * np.asarray([f0, f1], dtype=np.float32)).astype(np.float32)

        # Referee suppression: never map referee-like yellow/orange kits to team 0/1.
        def _is_referee_like(det_i: int) -> bool:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy[int(det_i)]]
            h, w = frame.shape[:2]
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))
            if (x2 - x1) < 10 or (y2 - y1) < 16:
                return False
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return False
            ch, cw = crop.shape[:2]
            jersey = crop[int(0.16 * ch) : int(0.62 * ch), int(0.18 * cw) : int(0.82 * cw)]
            if jersey.size == 0:
                return False
            hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
            hue = hsv[..., 0]
            sat = hsv[..., 1]
            val = hsv[..., 2]
            yellow_orange = (hue >= 10) & (hue <= 38) & (sat >= 100) & (val >= 145)
            ratio = float(np.mean(yellow_orange.astype(np.float32)))
            if ratio < 0.62:
                return False
            yo_h = hue[yellow_orange]
            yo_v = val[yellow_orange]
            if yo_h.size < 20:
                return False
            hue_std = float(np.std(yo_h.astype(np.float32)))
            val_mean = float(np.mean(yo_v.astype(np.float32)))
            # Neon ref shirts are usually very bright and uniform.
            return bool(hue_std <= 10.0 and val_mean >= 175.0)

        # Track-aware ref votes reduce one-frame mistakes while keeping team tagging stable.
        ref_votes = getattr(self, "_rt_ref_votes", None)
        if ref_votes is None:
            ref_votes = {}
            self._rt_ref_votes = ref_votes
        ref_seen = getattr(self, "_rt_ref_seen", None)
        if ref_seen is None:
            ref_seen = {}
            self._rt_ref_seen = ref_seen

        ref_candidates = []
        for i in player_idx:
            i = int(i)
            tid = _to_track_id(tracker_ids[i]) if tracker_ids is not None else None
            is_ref_now = _is_referee_like(i)
            if tid is not None:
                v = float(ref_votes.get(tid, 0.0))
                v = (0.84 * v) + (1.0 if is_ref_now else 0.0)
                ref_votes[tid] = v
                ref_seen[tid] = frame_idx
                is_ref_now = bool(v >= 1.15)
            if int(teams[i]) in (0, 1) and is_ref_now:
                ref_candidates.append(i)

        # Keep conservative for yellow-kit matches: only suppress small ref-like subset.
        max_refs = max(2, int(0.14 * len(player_idx)))
        if 1 <= len(ref_candidates) <= max_refs:
            for i in ref_candidates:
                teams[int(i)] = -1

        if tracker_ids is not None:
            for i in player_idx:
                tid = _to_track_id(tracker_ids[int(i)])
                _update_track_votes(self, tid, int(teams[int(i)]), frame_idx)

        _prune_track_votes(self, frame_idx)
        result["teams"] = teams

    compute_eval_bar = ctx.get("compute_eval_bar", None)

    def _estimate_player_voronoi_areas(
        positions: Any,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        nx: int = 42,
        ny: int = 28,
    ) -> Any:
        pos = np.asarray(positions, dtype=np.float32)
        if len(pos) == 0:
            return np.zeros((0,), dtype=np.float32)
        xs = np.linspace(0.0, float(pitch_length), int(nx), dtype=np.float32)
        ys = np.linspace(0.0, float(pitch_width), int(ny), dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        d = np.linalg.norm(grid[:, None, :] - pos[None, :, :], axis=2)
        owner = np.argmin(d, axis=1)
        counts = np.bincount(owner, minlength=len(pos)).astype(np.float32)
        cell_area = (float(pitch_length) / float(nx)) * (float(pitch_width) / float(ny))
        return counts * float(cell_area)

    def _team_relative_xt(x: float, y: float, team_id: int, pitch_length: float) -> float:
        # Team-relative xT lookup:
        # - team 0 uses base grid (high values on right)
        # - team 1 uses horizontally flipped grid (high values on left)
        xx = float(x)
        if int(team_id) == 1:
            xx = float(pitch_length) - float(xx)
        return float(get_xt(xx, float(y)))

    def _infer_attack_left_to_right(
        self: Any,
        pitch_coords: Any,
        teams: Any,
        class_ids: Any,
        possession_team: int,
    ) -> bool:
        prev = bool(getattr(self, "_rt_attack_left_to_right_prev", True))
        if possession_team not in (0, 1):
            return prev

        player_mask = (class_ids == self.cfg.PLAYER) | (class_ids == self.cfg.GOALKEEPER)
        if int(np.sum(player_mask)) < 4:
            return prev

        player_coords = np.asarray(pitch_coords[player_mask], dtype=np.float32)
        player_teams = np.asarray(teams[player_mask], dtype=int)
        team_mask = player_teams == int(possession_team)
        if int(np.sum(team_mask)) < 2:
            return prev

        team_x = player_coords[team_mask, 0]
        half_x = float(getattr(self.cfg, "pitch_length", 105.0)) * 0.5
        left_n = int(np.sum(team_x < half_x))
        right_n = int(np.sum(team_x >= half_x))
        if left_n == right_n:
            return prev

        # User rule: majority right => attacking right (base grid), majority left => attacking left (flip grid).
        attack_left_to_right = bool(right_n > left_n)
        self._rt_attack_left_to_right_prev = attack_left_to_right
        return attack_left_to_right

    def _apply_eval_xt_orientation(self: Any, result: Dict[str, Any]) -> None:
        ball_pos = result.get("ball_pos")
        if ball_pos is None:
            return
        try:
            bx = float(ball_pos[0])
            by = float(ball_pos[1])
        except Exception:
            return

        pitch_length = float(getattr(self.cfg, "pitch_length", 105.0))

        ball_xt_team0 = _team_relative_xt(bx, by, 0, pitch_length)
        ball_xt_team1 = _team_relative_xt(bx, by, 1, pitch_length)
        result["_rt_ball_xt_team0"] = float(ball_xt_team0)
        result["_rt_ball_xt_team1"] = float(ball_xt_team1)

        xt_neutral = float(ctx.get("XT_NEUTRAL", 0.018))
        xt_spread = max(float(ctx.get("XT_SPREAD", 0.032)), 1e-6)
        xt_min = float(ctx.get("XT_MIN", 0.006))
        xt_max = float(ctx.get("XT_MAX", 0.124))
        xt_den = max(xt_max - xt_min, 1e-6)
        xt0_centered = float(np.tanh((ball_xt_team0 - xt_neutral) / xt_spread))
        xt1_centered = float(np.tanh((ball_xt_team1 - xt_neutral) / xt_spread))
        xt_adv = float(np.clip(xt0_centered - xt1_centered, -1.0, 1.0))
        result["_rt_xt_adv_team0_minus_team1"] = xt_adv

        # Eval uses team-relative threat advantage (team0 perspective), not raw shared xT.
        possession_team = int(result.get("possession_team", -1))
        pitch_control = result.get("pitch_control", {0: 0.5, 1: 0.5})
        possession_pct = float(result.get("possession_pct", 0.5))
        pressure = float(result.get("pressure_diff", 0.0))
        reliability = float(result.get("signal_reliability", 1.0))
        pc_diff = float(np.clip(pitch_control.get(0, 0.5) - pitch_control.get(1, 0.5), -1.0, 1.0))
        poss_diff = float(np.clip((possession_pct - 0.5) * 2.0, -1.0, 1.0))
        press_diff = float(np.clip(pressure, -1.0, 1.0))

        # Dribble / carry momentum bonus:
        # if the ball's team-relative xT increases frame-over-frame while possession is held
        # (or briefly lost due to detector uncertainty), add an extra boost so sustained runs
        # toward goal keep climbing.
        prev_team = int(getattr(self, "_rt_eval_prev_possession_team", -1))
        prev_active_xt = getattr(self, "_rt_eval_prev_active_team_xt", None)
        mom_state = float(getattr(self, "_rt_eval_dribble_momentum_state", 0.0))
        dribble_bonus = 0.0
        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        pos_hold_team = int(getattr(self, "_rt_eval_bonus_poss_team", -1))
        pos_hold_ttl = int(getattr(self, "_rt_eval_bonus_poss_ttl", 0))
        if possession_team in (0, 1):
            eff_team = int(possession_team)
            pos_hold_team = eff_team
            pos_hold_ttl = 10
        elif pos_hold_team in (0, 1) and pos_hold_ttl > 0 and ball_pos is not None:
            # Carry forward last reliable possession team across short detector dropouts.
            eff_team = int(pos_hold_team)
            pos_hold_ttl -= 1
        else:
            eff_team = -1
            pos_hold_team = -1
            pos_hold_ttl = 0

        self._rt_eval_bonus_poss_team = int(pos_hold_team)
        self._rt_eval_bonus_poss_ttl = int(max(0, pos_hold_ttl))

        active_xt_norm = 0.0
        if eff_team in (0, 1):
            active_xt = float(ball_xt_team0 if eff_team == 0 else ball_xt_team1)
            active_xt_norm = float(np.clip((active_xt - xt_min) / xt_den, 0.0, 1.0))
            if prev_team == eff_team and prev_active_xt is not None:
                delta_xt = float(active_xt - float(prev_active_xt))
                if delta_xt > 0.0:
                    delta_norm = float(np.clip(delta_xt / xt_den, 0.0, 1.0))
                    # Build a momentum state that grows faster as the carrier enters dangerous zones.
                    growth_gain = 0.65 + 1.55 * active_xt_norm
                    if scenario_name == "martial":
                        growth_gain = 0.90 + 2.20 * active_xt_norm
                    mom_state = float(0.88 * mom_state + delta_norm * growth_gain)
                else:
                    decay = 0.88 if scenario_name == "martial" else 0.80
                    mom_state = float(decay * mom_state)
            else:
                mom_state = float(0.0)

            # Exponential response for sustained positive progress.
            exp_k = 2.45 if scenario_name == "martial" else 1.85
            amp = 0.30 if scenario_name == "martial" else 0.16
            cap_bonus = 0.65 if scenario_name == "martial" else 0.42
            base_bonus = float(np.expm1(exp_k * max(0.0, mom_state)))
            signed = 1.0 if eff_team == 0 else -1.0
            dribble_bonus = float(signed * np.clip(amp * base_bonus, 0.0, cap_bonus))

            self._rt_eval_prev_possession_team = int(eff_team)
            self._rt_eval_prev_active_team_xt = float(active_xt)
        else:
            mom_state = float(0.90 * mom_state if scenario_name == "martial" else 0.65 * mom_state)
            self._rt_eval_prev_possession_team = -1
            self._rt_eval_prev_active_team_xt = None

        self._rt_eval_dribble_momentum_state = float(np.clip(mom_state, 0.0, 2.0))
        result["_rt_eval_dribble_bonus"] = float(dribble_bonus)

        if possession_team == -1:
            active_weights = dict(ctx.get("CONTESTED_EVAL_WEIGHTS", {"pitch_control": 0.46, "ball_value": 0.20, "possession": 0.22, "pressure": 0.12}))
        else:
            active_weights = dict(ctx.get("DEFAULT_EVAL_WEIGHTS", {"pitch_control": 0.34, "ball_value": 0.38, "possession": 0.16, "pressure": 0.12}))

        # Make xT advantage the dominant driver of the eval bar (user-requested).
        # If notebook defaults are lower, raise ball_value and shrink the others proportionally.
        min_ball_weight = 0.55
        ball_w = max(0.0, float(active_weights.get("ball_value", 0.0)))
        if ball_w < min_ball_weight:
            deficit = float(min_ball_weight - ball_w)
            other_keys = [k for k in ("pitch_control", "possession", "pressure") if float(active_weights.get(k, 0.0)) > 0.0]
            other_sum = float(sum(float(active_weights.get(k, 0.0)) for k in other_keys))
            if other_sum > 1e-9:
                shrink = min(deficit, other_sum - 1e-6)
                for k in other_keys:
                    wk = float(active_weights.get(k, 0.0))
                    active_weights[k] = max(0.0, wk - shrink * (wk / other_sum))
                active_weights["ball_value"] = float(ball_w + shrink)
            else:
                active_weights["ball_value"] = float(min_ball_weight)

        w_sum = float(sum(max(0.0, float(v)) for v in active_weights.values()))
        if w_sum > 1e-9:
            active_weights = {k: max(0.0, float(v)) / w_sum for k, v in active_weights.items()}

        linear = (
            active_weights.get("pitch_control", 0.0) * pc_diff
            + active_weights.get("ball_value", 0.0) * xt_adv
            + active_weights.get("possession", 0.0) * poss_diff
            + active_weights.get("pressure", 0.0) * press_diff
            + dribble_bonus
        )

        # Martial-only skew: attacking holder Voronoi area * xT * 3 (user-requested "original" feel).
        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        if scenario_name == "martial" and eff_team in (0, 1):
            holder_area_norm = float(np.clip(result.get("_rt_holder_voronoi_area_norm", 0.0), 0.0, 1.0))
            skew_mag = float(np.clip(3.0 * holder_area_norm * active_xt_norm, 0.0, 1.25))
            linear += (1.0 if eff_team == 0 else -1.0) * skew_mag
            result["_rt_martial_skew_bonus"] = float((1.0 if eff_team == 0 else -1.0) * skew_mag)
        else:
            result["_rt_martial_skew_bonus"] = 0.0
        # So low-confidence frames still move meaningfully instead of flattening spikes.
        rel = float(np.clip(reliability, 0.40, 1.0))
        result["eval_bar"] = float(np.clip(100.0 * np.tanh(1.60 * rel * linear), -100.0, 100.0))

    def _strict_holder_from_detected_ball(self: Any, result: Dict[str, Any]) -> None:
        """Strict holder: only from current detected ball (no previous-holder fallback)."""
        _os = __import__("os")
        debug_s3 = bool(int(_os.environ.get("RT_DEBUG_SCENARIO3_PASS", "0") or "0"))
        scenario_name_dbg = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        frame_idx_dbg = int(getattr(self, "_rt_frame_idx", 0))

        def _dbg(msg: str) -> None:
            if debug_s3 and scenario_name_dbg == "scenario3" and frame_idx_dbg < 100:
                print(f"[s3-passdbg f={frame_idx_dbg}] {msg}")

        def _fallback_ball_px_from_color_near_players(class_ids_arr: Any) -> Optional[np.ndarray]:
            sn = str(getattr(self, "_rt_scenario_name", "") or "").lower()
            if sn not in ("scenario3", "giroud", "scenario5"):
                return None
            try:
                p_idx = np.where((class_ids_arr == self.cfg.PLAYER) | (class_ids_arr == self.cfg.GOALKEEPER))[0]
                if len(p_idx) == 0:
                    return None
                p_boxes = np.asarray(det.xyxy[p_idx], dtype=np.float32)
                p_anchors = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[p_idx]
                hsvf = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h = hsvf[..., 0]
                s = hsvf[..., 1]
                v = hsvf[..., 2]
                # Old EPL clip ball is orange/yellow. Keep mask narrow to avoid jerseys.
                warm = (((h >= 8) & (h <= 30) & (s >= 95) & (v >= 95))).astype(np.uint8) * 255
                warm = cv2.medianBlur(warm, 3)
                contours, _ = cv2.findContours(warm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_pt = None
                best_score = 1e9
                for c in contours:
                    area = float(cv2.contourArea(c))
                    if area < 6.0 or area > 220.0:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    cx = float(x + 0.5 * w)
                    cy = float(y + 0.5 * h)
                    # Must be near some player's lower body region.
                    for i, (bx1, by1, bx2, by2) in enumerate(p_boxes):
                        if not (bx1 - 18 <= cx <= bx2 + 18 and by1 + 0.35 * (by2 - by1) <= cy <= by2 + 28):
                            continue
                        d = float(np.linalg.norm(np.asarray(p_anchors[i], dtype=np.float32) - np.asarray([cx, cy], dtype=np.float32)))
                        if d > 78.0:
                            continue
                        # Prefer closer-to-foot and smaller compact blobs.
                        score = d + 0.05 * area
                        if score < best_score:
                            best_score = score
                            best_pt = np.asarray([cx, cy], dtype=np.float32)
                return best_pt
            except Exception:
                return None

        det = result.get("detections")
        pitch_coords = result.get("pitch_coords")
        teams = result.get("teams")
        if det is None or pitch_coords is None or teams is None:
            _dbg("missing det/pitch/teams")
            result["_rt_attack_left_to_right"] = bool(getattr(self, "_rt_attack_left_to_right_prev", True))
            result["pass_options"] = []
            result["_rt_ball_detected"] = False
            result["_rt_holder_confident"] = False
            result["_rt_best_pass"] = None
            result["_rt_holder_voronoi_area_norm"] = 0.0
            result["_rt_ball_holder_idx"] = -1
            return

        class_ids = np.asarray(det.class_id)
        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        proxy_ball_pitch_pre = result.get("ball_pos", None)
        ball_idx = np.where(class_ids == self.cfg.BALL)[0]
        _dbg(f"balls={len(ball_idx)}")
        use_pitch_proxy_ball = False
        ball_px = None
        ball_pitch = None
        if len(ball_idx) == 0:
            if scenario_name in ("scenario3", "giroud", "scenario5"):
                try:
                    pb = np.asarray(proxy_ball_pitch_pre, dtype=np.float32) if proxy_ball_pitch_pre is not None else None
                except Exception:
                    pb = None
                if pb is not None and pb.shape == (2,) and np.isfinite(pb).all():
                    use_pitch_proxy_ball = True
                    ball_pitch = pb
                    result["_rt_ball_pixel"] = None
                    result["_rt_ball_detected"] = True
                    result["_rt_ball_proxy_used"] = True
                    result["ball_pos"] = ball_pitch
                    _dbg(f"use_pitch_proxy_ball pitch=({ball_pitch[0]:.2f},{ball_pitch[1]:.2f})")
                if not use_pitch_proxy_ball:
                    fb_px = _fallback_ball_px_from_color_near_players(class_ids)
                    if fb_px is not None:
                        ball_px = fb_px.astype(np.float32)
                        use_pitch_proxy_ball = False
                        result["_rt_ball_pixel"] = ball_px
                        result["_rt_ball_detected"] = True
                        result["_rt_ball_proxy_used"] = True
                        _dbg(f"use_color_ball_px_proxy px=({ball_px[0]:.1f},{ball_px[1]:.1f})")
            if not use_pitch_proxy_ball:
                if ball_px is not None:
                    # continue via pixel-ball path below
                    pass
                else:
                    _dbg("no_ball_detection")
                    result["_rt_attack_left_to_right"] = _infer_attack_left_to_right(
                        self,
                        pitch_coords,
                        np.asarray(teams, dtype=int),
                        class_ids,
                        int(result.get("possession_team", -1)),
                    )
                    result["ball_pos"] = None
                    result["pass_options"] = []
                    result["_rt_ball_detected"] = False
                    result["_rt_ball_proxy_used"] = False
                    result["_rt_holder_confident"] = False
                    result["_rt_best_pass"] = None
                    result["_rt_holder_voronoi_area_norm"] = 0.0
                    result["_rt_ball_holder_idx"] = -1
                    return
        if not use_pitch_proxy_ball:
            # Choose most confident current ball detection.
            if ball_px is None and getattr(det, "confidence", None) is not None:
                conf = np.asarray(det.confidence, dtype=np.float32)
                b_local = int(np.argmax(conf[ball_idx]))
                b_idx = int(ball_idx[b_local])
            elif ball_px is None:
                b_idx = int(ball_idx[0])
            if ball_px is None:
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[b_idx]]
                ball_px = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
                result["_rt_ball_pixel"] = ball_px
                result["_rt_ball_detected"] = True
                result["_rt_ball_proxy_used"] = False
                _dbg(f"ball_px=({ball_px[0]:.1f},{ball_px[1]:.1f}) bidx={b_idx}")

                # Keep true detected ball pitch position.
                ball_pitch = np.asarray(pitch_coords[b_idx], dtype=np.float32)
                result["ball_pos"] = ball_pitch

        player_idx = np.where((class_ids == self.cfg.PLAYER) | (class_ids == self.cfg.GOALKEEPER))[0]
        _dbg(f"players={len(player_idx)}")
        if len(player_idx) == 0:
            _dbg("no_players")
            result["_rt_attack_left_to_right"] = _infer_attack_left_to_right(
                self,
                pitch_coords,
                np.asarray(teams, dtype=int),
                class_ids,
                int(result.get("possession_team", -1)),
            )
            result["pass_options"] = []
            result["_rt_holder_confident"] = False
            result["_rt_best_pass"] = None
            result["_rt_holder_voronoi_area_norm"] = 0.0
            result["_rt_ball_holder_idx"] = -1
            return

        player_pixels = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)[player_idx]
        player_boxes = np.asarray(det.xyxy[player_idx], dtype=np.float32)
        player_pitch = np.asarray(pitch_coords[player_idx], dtype=np.float32)
        player_teams = np.asarray(np.asarray(teams)[player_idx], dtype=int)

        # Rule 1: holder bbox contains the detected ball center.
        contained = []
        if ball_px is not None:
            for i, (bx1, by1, bx2, by2) in enumerate(player_boxes):
                pad = 3.0
                if (bx1 - pad) <= ball_px[0] <= (bx2 + pad) and (by1 - pad) <= ball_px[1] <= (by2 + pad):
                    contained.append(i)

        holder_local = -1
        holder_team = -1
        confident = False

        if len(contained) > 0:
            # If multiple, nearest bottom-center anchor wins.
            d = np.linalg.norm(player_pixels[np.asarray(contained)] - ball_px[None, :], axis=1)
            holder_local = int(contained[int(np.argmin(d))])
            holder_team = int(player_teams[holder_local])
            confident = holder_team in (0, 1)
            _dbg(f"holder_from_contained local={holder_local} team={holder_team} n_contained={len(contained)} confident={confident}")
        elif ball_px is not None:
            # Rule 2: nearest player to detected ball center, but only if close enough.
            d = np.linalg.norm(player_pixels - ball_px[None, :], axis=1)
            if len(d) > 0:
                d_sorted = np.sort(d)
                k = int(np.argmin(d))
                nearest = float(d[k])
                second_gap = float(d_sorted[1] - d_sorted[0]) if len(d_sorted) > 1 else 99.0
                # Strict confidence threshold in pixels + separation from second nearest.
                max_nearest_px = 30.0
                min_gap_px = 4.0
                if scenario_name in ("scenario3", "giroud", "scenario5"):
                    max_nearest_px = 62.0
                    min_gap_px = 3.0
                if nearest <= max_nearest_px and second_gap >= min_gap_px:
                    holder_local = k
                    holder_team = int(player_teams[k])
                    confident = holder_team in (0, 1)
                _dbg(f"holder_from_nearest local={holder_local} team={holder_team} nearest={nearest:.1f} gap={second_gap:.1f} confident={confident}")
        else:
            # Scenario3/giroud fallback path: derive holder from upstream pitch-space ball proxy.
            d_pitch = np.linalg.norm(player_pitch - np.asarray(ball_pitch, dtype=np.float32)[None, :], axis=1)
            if len(d_pitch) > 0:
                order = np.argsort(d_pitch)
                k = int(order[0])
                nearest_m = float(d_pitch[k])
                second_gap_m = float(d_pitch[order[1]] - d_pitch[order[0]]) if len(order) > 1 else 99.0
                if nearest_m <= 5.2 and second_gap_m >= 0.45:
                    holder_local = k
                    holder_team = int(player_teams[k])
                    confident = holder_team in (0, 1)
                _dbg(
                    f"holder_from_pitch_proxy local={holder_local} team={holder_team} "
                    f"nearest_m={nearest_m:.2f} gap_m={second_gap_m:.2f} confident={confident}"
                )

        result["_rt_ball_holder_idx"] = int(holder_local)
        result["_rt_holder_confident"] = bool(confident)

        if not confident or holder_local < 0:
            _dbg(f"holder_fail confident={confident} holder_local={holder_local}")
            result["_rt_attack_left_to_right"] = _infer_attack_left_to_right(
                self,
                pitch_coords,
                np.asarray(teams, dtype=int),
                class_ids,
                int(result.get("possession_team", -1)),
            )
            result["pass_options"] = []
            result["_rt_best_pass"] = None
            result["possession_team"] = -1
            result["_rt_holder_voronoi_area_norm"] = 0.0
            return

        holder_pos = player_pitch[int(holder_local)]
        result["_rt_ball_holder_pos"] = holder_pos
        result["possession_team"] = int(holder_team)
        attack_left_to_right = _infer_attack_left_to_right(
            self,
            pitch_coords,
            np.asarray(teams, dtype=int),
            class_ids,
            int(holder_team),
        )
        result["_rt_attack_left_to_right"] = bool(attack_left_to_right)

        # Pass options are only for this strict holder.
        raw_opts = ctx["find_pass_options"](holder_pos, int(holder_team), player_pitch, player_teams)
        _dbg(f"holder_ok team={holder_team} raw_opts={len(raw_opts)}")

        scenario_name = str(getattr(self, "_rt_scenario_name", "") or "").lower()
        if scenario_name == "martial":
            xt_weight = 0.92
            area_weight = 0.08
        else:
            xt_weight = 0.70
            area_weight = 0.30

        # Optimal pass score = xt_weight*xT_norm + area_weight*Voronoi_area_norm
        areas = _estimate_player_voronoi_areas(
            player_pitch,
            pitch_length=float(getattr(self.cfg, "pitch_length", 105.0)),
            pitch_width=float(getattr(self.cfg, "pitch_width", 68.0)),
            nx=42,
            ny=28,
        )
        mates = np.where(player_teams == int(holder_team))[0]
        if len(mates) == 0:
            _dbg("no_teammates_after_filter")
            result["pass_options"] = []
            result["_rt_best_pass"] = None
            result["_rt_holder_voronoi_area_norm"] = 0.0
            return
        mate_areas = areas[mates] if len(areas) == len(player_pitch) else np.zeros((len(mates),), dtype=np.float32)
        a_min = float(np.min(mate_areas)) if len(mate_areas) > 0 else 0.0
        a_max = float(np.max(mate_areas)) if len(mate_areas) > 0 else 1.0
        a_den = max(a_max - a_min, 1e-6)
        holder_area = float(areas[int(holder_local)]) if int(holder_local) < len(areas) else 0.0
        holder_area_norm = float(np.clip((holder_area - a_min) / a_den, 0.0, 1.0))
        result["_rt_holder_voronoi_area_m2"] = holder_area
        result["_rt_holder_voronoi_area_norm"] = holder_area_norm
        xt_min = float(ctx.get("XT_MIN", 0.006))
        xt_max = float(ctx.get("XT_MAX", 0.124))
        xt_den = max(xt_max - xt_min, 1e-6)

        best = None
        best_score = -1.0
        for opt in raw_opts:
            t = np.asarray(opt.get("target_pos"), dtype=np.float32)
            if t.shape != (2,):
                continue
            d = np.linalg.norm(player_pitch - t[None, :], axis=1)
            ridx = int(np.argmin(d))
            rec_area = float(areas[ridx]) if ridx < len(areas) else 0.0
            area_norm = float(np.clip((rec_area - a_min) / a_den, 0.0, 1.0))
            end_xt = _team_relative_xt(
                float(t[0]),
                float(t[1]),
                int(holder_team),
                float(getattr(self.cfg, "pitch_length", 105.0)),
            )
            xt_norm = float(np.clip((end_xt - xt_min) / xt_den, 0.0, 1.0))
            score = float((xt_weight * xt_norm) + (area_weight * area_norm))
            opt["optimal_score"] = float(np.clip(score, 0.0, 1.0))
            opt["receiver_area_m2"] = rec_area
            opt["xt_target"] = end_xt
            opt["is_best"] = False
            if opt["optimal_score"] > best_score:
                best_score = opt["optimal_score"]
                best = opt

        if best is not None:
            best["is_best"] = True

        result["pass_options"] = raw_opts
        result["_rt_best_pass"] = best
        _dbg(f"pass_options_final={len(raw_opts)} best_score={(best_score if best is not None else -1):.3f}")

    def _risk_color(opt: Dict[str, Any], is_worst: bool = False) -> tuple[int, int, int]:
        if bool(opt.get("is_best", False)):
            return (0, 255, 0)  # green best
        if is_worst:
            return (0, 0, 255)  # red least-optimal
        return (0, 255, 255)  # yellow medium/safe

    def _select_display_pass_options(holder_pos: Any, pass_options: Any, max_n: int = 6) -> list[Dict[str, Any]]:
        opts = [o for o in list(pass_options) if isinstance(o, dict)]
        if holder_pos is None or len(opts) == 0:
            return opts[:max_n]
        hp = np.asarray(holder_pos, dtype=np.float32)
        if hp.shape != (2,):
            return opts[:max_n]

        # Rank helpers.
        def _dist(o: Dict[str, Any]) -> float:
            t = np.asarray(o.get("target_pos"), dtype=np.float32)
            if t.shape != (2,):
                return 1e9
            return float(np.linalg.norm(t - hp))

        by_score = sorted(opts, key=lambda o: float(o.get("optimal_score", 0.0)), reverse=True)
        by_near = sorted(opts, key=_dist)
        worst = min(opts, key=lambda o: float(o.get("optimal_score", 1.0)))

        picked: list[Dict[str, Any]] = []
        seen_ids = set()

        def _push(o: Optional[Dict[str, Any]]) -> None:
            if o is None:
                return
            oid = id(o)
            if oid in seen_ids:
                return
            seen_ids.add(oid)
            picked.append(o)

        # Guarantee best + nearest teammates + one risky option for contrast.
        if by_score:
            _push(by_score[0])
        for o in by_near[:2]:
            _push(o)
        _push(worst)
        for o in by_score[1:]:
            if len(picked) >= max_n:
                break
            _push(o)
        return picked[:max_n]

    def _create_xt_heatmap_oriented(width: int, height: int, flip_lr: bool) -> Any:
        heatmap = np.zeros((height, width, 3), dtype=np.uint8)
        cell_w = width / 12.0
        cell_h = height / 8.0
        grid = np.fliplr(XT_GRID) if bool(flip_lr) else XT_GRID
        xt_norm = np.clip(grid / 0.15, 0.0, 1.0)

        for row in range(8):
            for col in range(12):
                x1 = int(col * cell_w)
                y1 = int(row * cell_h)
                x2 = int((col + 1) * cell_w)
                y2 = int((row + 1) * cell_h)
                val = float(xt_norm[row, col])
                r = int(255 * val)
                b = int(255 * (1.0 - val))
                g = int(100 * (1.0 - abs(val - 0.5) * 2.0))
                cv2.rectangle(heatmap, (x1, y1), (x2, y2), (b, g, r), -1)
                txt = f"{float(grid[row, col]):.3f}"
                cv2.putText(heatmap, txt, (x1 + 5, y1 + int(cell_h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return heatmap

    def _draw_pass_lanes_on_pitch_patched(
        pitch: Any,
        ball_holder_pos: Any,
        pass_options: Any,
        width: int = 600,
        height: int = 400,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        alpha: float = 1.0,
    ) -> Any:
        if ball_holder_pos is None or len(pass_options) == 0:
            return pitch
        sx = float(width) / float(pitch_length)
        sy = float(height) / float(pitch_width)
        bx = int(float(ball_holder_pos[0]) * sx)
        by = int(float(ball_holder_pos[1]) * sy)

        opts = _select_display_pass_options(ball_holder_pos, pass_options, max_n=6)
        worst = min(opts, key=lambda p: float(p.get("optimal_score", 1.0))) if len(opts) > 0 else None
        draw_img = pitch if float(alpha) >= 0.999 else pitch.copy()

        for opt in opts:
            t = np.asarray(opt.get("target_pos"), dtype=np.float32)
            if t.shape != (2,):
                continue
            tx = int(float(t[0]) * sx)
            ty = int(float(t[1]) * sy)
            is_worst = bool(worst is opt)
            color = _risk_color(opt, is_worst=is_worst)
            thick = 4 if bool(opt.get("is_best", False)) else 2
            cv2.line(draw_img, (bx, by), (tx, ty), color, thick)
            cv2.circle(draw_img, (tx, ty), 7 if bool(opt.get("is_best", False)) else 5, color, -1)

        cv2.circle(draw_img, (bx, by), 12, (0, 255, 255), 3)
        if float(alpha) < 0.999:
            a = float(np.clip(alpha, 0.0, 1.0))
            cv2.addWeighted(draw_img, a, pitch, 1.0 - a, 0.0, dst=pitch)
        return pitch

    def _process_video_multi_output_patched(
        video_path: str,
        output_dir: str,
        analyzer: Any,
        max_frames: int = None,
        skip_frames: int = 2,
        start_frame: int = 0,
    ):
        Path = ctx["Path"]
        draw_eval_bar = ctx["draw_eval_bar"]
        create_voronoi_pitch = ctx["create_voronoi_pitch"]
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        cap = cv2.VideoCapture(video_path)
        total_frames_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = max(0, int(start_frame))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        total_frames = max(0, total_frames_all - start_frame)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        output_fps = fps / max(1, int(skip_frames))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_eval = cv2.VideoWriter(f"{output_dir}/eval_overlay.mp4", fourcc, output_fps, (width, height))
        out_pass = cv2.VideoWriter(f"{output_dir}/pass_prediction.mp4", fourcc, output_fps, (width, height))
        tactical_w, tactical_h = 600, 400
        combined_w = width + tactical_w
        combined_h = max(height, tactical_h)
        out_tactical = cv2.VideoWriter(f"{output_dir}/tactical_view.mp4", fourcc, output_fps, (combined_w, combined_h))

        print(f"Processing {total_frames} frames into 3 outputs...")
        print(f"  1. {output_dir}/eval_overlay.mp4")
        print(f"  2. {output_dir}/pass_prediction.mp4")
        print(f"  3. {output_dir}/tactical_view.mp4")

        frame_idx = 0
        processed = 0
        last_tactical_pitch = None
        pass_persist_frames = max(1, int(round(float(output_fps) * 0.5)))
        pass_fade_frames = max(1, min(5, pass_persist_frames))
        pass_cache: Optional[Dict[str, Any]] = None

        def _cache_clone_options(pass_options: Any) -> list[dict[str, Any]]:
            cloned = []
            for opt in list(pass_options)[:6]:
                c = dict(opt)
                t = c.get("target_pos")
                if t is not None:
                    c["target_pos"] = np.asarray(t, dtype=np.float32).copy()
                cloned.append(c)
            return cloned

        def _cache_build_broadcast_snapshot(holder_px: Any, player_pixels: Any, player_coords: Any, pass_options: Any) -> Optional[Dict[str, Any]]:
            if holder_px is None or len(pass_options) == 0:
                return None
            opts = _select_display_pass_options(holder_px, pass_options, max_n=6)
            worst = min(opts, key=lambda p: float(p.get("optimal_score", 1.0))) if len(opts) > 0 else None
            segs = []
            holder_px_arr = np.asarray(holder_px, dtype=np.float32).copy()
            player_pixels_arr = np.asarray(player_pixels, dtype=np.float32)
            player_coords_arr = np.asarray(player_coords, dtype=np.float32)
            for opt in opts:
                target_pitch = np.asarray(opt.get("target_pos"), dtype=np.float32)
                if target_pitch.shape != (2,) or len(player_coords_arr) == 0:
                    continue
                dists = np.linalg.norm(player_coords_arr - target_pitch[None, :], axis=1)
                closest_idx = int(np.argmin(dists))
                if float(dists[closest_idx]) > 3.0:
                    continue
                target_px = np.asarray(player_pixels_arr[closest_idx], dtype=np.float32).copy()
                segs.append(
                    {
                        "target_px": target_px,
                        "color": _risk_color(opt, is_worst=bool(worst is opt)),
                        "thick": 5 if bool(opt.get("is_best", False)) else 3,
                        "dot_r": 9 if bool(opt.get("is_best", False)) else 7,
                    }
                )
            if len(segs) == 0:
                return None
            return {"holder_px": holder_px_arr, "segments": segs}

        def _draw_cached_pass_overlay(frame_img: Any, snap: Optional[Dict[str, Any]], alpha: float = 1.0) -> Any:
            if snap is None:
                return frame_img
            a = float(np.clip(alpha, 0.0, 1.0))
            holder = np.asarray(snap.get("holder_px"), dtype=np.float32)
            if holder.shape != (2,):
                return frame_img
            draw_img = frame_img if a >= 0.999 else frame_img.copy()
            pt1 = tuple(map(int, holder))
            for seg in snap.get("segments", []):
                target_px = np.asarray(seg.get("target_px"), dtype=np.float32)
                if target_px.shape != (2,):
                    continue
                pt2 = tuple(map(int, target_px))
                cv2.line(draw_img, pt1, pt2, tuple(seg.get("color", (0, 255, 255))), int(seg.get("thick", 3)))
                cv2.circle(draw_img, pt2, int(seg.get("dot_r", 7)), tuple(seg.get("color", (0, 255, 255))), -1)
            cv2.circle(draw_img, pt1, 20, (0, 255, 255), 3)
            if a < 0.999:
                cv2.addWeighted(draw_img, a, frame_img, 1.0 - a, 0.0, dst=frame_img)
            return frame_img

        def _cache_alpha(cache: Optional[Dict[str, Any]]) -> float:
            if cache is None:
                return 0.0
            rem = int(cache.get("remaining", 0))
            if rem <= 0:
                return 0.0
            if rem > pass_fade_frames:
                return 1.0
            return float(np.clip(rem / float(pass_fade_frames), 0.15, 1.0))

        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(1, int(skip_frames)) != 0:
                frame_idx += 1
                continue

            result = analyzer.process_frame(frame)

            # OUTPUT 1: eval overlay
            eval_frame = frame.copy()
            if result.get("detections") is not None:
                det = result["detections"]
                teams = result["teams"] if result.get("teams") is not None else np.full(len(det), -1)
                for i, (xyxy, cls_id) in enumerate(zip(det.xyxy, det.class_id)):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2) // 2, y2
                    if cls_id == analyzer.cfg.BALL:
                        cv2.circle(eval_frame, (cx, cy), 10, BALL_COLOR, -1)
                        cv2.circle(eval_frame, (cx, cy), 10, (0, 0, 0), 2)
                    elif cls_id in (analyzer.cfg.PLAYER, analyzer.cfg.GOALKEEPER):
                        color = TEAM_COLORS.get(int(teams[i]), TEAM_COLORS[-1])
                        cv2.circle(eval_frame, (cx, cy), 14, color, -1)
                        cv2.circle(eval_frame, (cx, cy), 14, (0, 0, 0), 2)
            eval_frame = draw_eval_bar(eval_frame, result.get("eval_bar", 0.0), pos=(50, 50), size=(300, 40))
            out_eval.write(eval_frame)

            # OUTPUT 2: strict pass rendering
            pass_frame = frame.copy()
            current_pass_cache: Optional[Dict[str, Any]] = None
            used_cached_pass = False
            if result.get("detections") is not None and bool(result.get("homography_valid", False)):
                det = result["detections"]
                teams = result["teams"] if result.get("teams") is not None else np.full(len(det), -1)
                pixels = det.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

                for i, (xyxy, cls_id) in enumerate(zip(det.xyxy, det.class_id)):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2) // 2, y2
                    if cls_id == analyzer.cfg.BALL:
                        cv2.circle(pass_frame, (cx, cy), 10, BALL_COLOR, -1)
                    elif cls_id in (analyzer.cfg.PLAYER, analyzer.cfg.GOALKEEPER):
                        color = TEAM_COLORS.get(int(teams[i]), TEAM_COLORS[-1])
                        cv2.circle(pass_frame, (cx, cy), 14, color, -1)
                        cv2.circle(pass_frame, (cx, cy), 14, (0, 0, 0), 2)

                player_mask = (det.class_id == analyzer.cfg.PLAYER) | (det.class_id == analyzer.cfg.GOALKEEPER)
                player_pixels = np.asarray(pixels[player_mask], dtype=np.float32)
                player_coords = np.asarray(result["pitch_coords"][player_mask], dtype=np.float32)
                opts = list(result.get("pass_options", []))
                holder_idx = int(result.get("_rt_ball_holder_idx", -1))

                if bool(result.get("_rt_ball_detected", False)) and bool(result.get("_rt_holder_confident", False)) and holder_idx >= 0 and holder_idx < len(player_pixels) and len(opts) > 0:
                    ball_holder_pixel = player_pixels[holder_idx]
                    snap = _cache_build_broadcast_snapshot(ball_holder_pixel, player_pixels, player_coords, opts)
                    if snap is not None:
                        current_pass_cache = {
                            "remaining": pass_persist_frames,
                            "broadcast": snap,
                            "holder_pitch": np.asarray(player_coords[holder_idx], dtype=np.float32).copy(),
                            "pass_options": _cache_clone_options(opts),
                        }
                        _draw_cached_pass_overlay(pass_frame, snap, alpha=1.0)

            if current_pass_cache is None and pass_cache is not None and int(pass_cache.get("remaining", 0)) > 0:
                _draw_cached_pass_overlay(pass_frame, pass_cache.get("broadcast"), alpha=_cache_alpha(pass_cache))
                used_cached_pass = True

            cv2.putText(pass_frame, "Pass Risk:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.circle(pass_frame, (180, 45), 8, (0, 255, 0), -1)
            cv2.putText(pass_frame, "Best", (195, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(pass_frame, (260, 45), 8, (0, 255, 255), -1)
            cv2.putText(pass_frame, "Med/Safe", (275, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.circle(pass_frame, (390, 45), 8, (0, 0, 255), -1)
            cv2.putText(pass_frame, "Risky", (405, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            out_pass.write(pass_frame)

            # OUTPUT 3: tactical view
            tactical_combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
            tactical_combined[:] = (30, 30, 30)
            broadcast_resized = cv2.resize(frame, (width, height))
            if result.get("detections") is not None:
                det = result["detections"]
                teams = result["teams"] if result.get("teams") is not None else np.full(len(det), -1)
                for i, (xyxy, cls_id) in enumerate(zip(det.xyxy, det.class_id)):
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2) // 2, y2
                    if cls_id == analyzer.cfg.BALL:
                        cv2.circle(broadcast_resized, (cx, cy), 8, BALL_COLOR, -1)
                    elif cls_id in (analyzer.cfg.PLAYER, analyzer.cfg.GOALKEEPER):
                        color = TEAM_COLORS.get(int(teams[i]), TEAM_COLORS[-1])
                        cv2.circle(broadcast_resized, (cx, cy), 10, color, -1)
            tactical_combined[0:height, 0:width] = broadcast_resized

            if result.get("pitch_coords") is not None and result.get("teams") is not None:
                player_mask = (result["detections"].class_id == analyzer.cfg.PLAYER) | (result["detections"].class_id == analyzer.cfg.GOALKEEPER)
                # Keep full arrays for holder indexing, but exclude non-team players
                # (e.g., referee mapped to -1) from Voronoi/pitch-control polygons.
                player_coords_all = np.asarray(result["pitch_coords"][player_mask], dtype=np.float32)
                player_teams_all = np.asarray(result["teams"][player_mask], dtype=int)
                team_mask = np.isin(player_teams_all, [0, 1])
                voronoi_coords = player_coords_all[team_mask]
                voronoi_teams = player_teams_all[team_mask]

                tactical_pitch = create_voronoi_pitch(
                    voronoi_coords,
                    voronoi_teams,
                    result.get("ball_pos"),
                    width=tactical_w,
                    height=tactical_h,
                    show_xt=False,
                    show_areas=True,
                )

                # Display xT from current possession team's perspective.
                poss_team = int(result.get("possession_team", -1))
                if poss_team == 1:
                    flip_lr = True
                elif poss_team == 0:
                    flip_lr = False
                else:
                    attack_ltr = bool(result.get("_rt_attack_left_to_right", True))
                    flip_lr = not attack_ltr
                xt_overlay = _create_xt_heatmap_oriented(tactical_w, tactical_h, flip_lr=flip_lr)
                tactical_pitch = cv2.addWeighted(tactical_pitch, 0.82, xt_overlay, 0.28, 0)

                holder_idx = int(result.get("_rt_ball_holder_idx", -1))
                if current_pass_cache is not None:
                    tactical_pitch = _draw_pass_lanes_on_pitch_patched(
                        tactical_pitch,
                        current_pass_cache.get("holder_pitch"),
                        current_pass_cache.get("pass_options", []),
                        width=tactical_w,
                        height=tactical_h,
                        alpha=1.0,
                    )
                elif pass_cache is not None and int(pass_cache.get("remaining", 0)) > 0:
                    tactical_pitch = _draw_pass_lanes_on_pitch_patched(
                        tactical_pitch,
                        pass_cache.get("holder_pitch"),
                        pass_cache.get("pass_options", []),
                        width=tactical_w,
                        height=tactical_h,
                        alpha=_cache_alpha(pass_cache),
                    )
                    used_cached_pass = True
                elif bool(result.get("_rt_ball_detected", False)) and bool(result.get("_rt_holder_confident", False)) and holder_idx >= 0 and holder_idx < len(player_coords_all) and len(result.get("pass_options", [])) > 0:
                    tactical_pitch = _draw_pass_lanes_on_pitch_patched(
                        tactical_pitch,
                        player_coords_all[holder_idx],
                        result["pass_options"],
                        width=tactical_w,
                        height=tactical_h,
                    )

                tactical_pitch = draw_eval_bar(tactical_pitch, result.get("eval_bar", 0.0), pos=(tactical_w - 220, 10), size=(200, 25))
                last_tactical_pitch = tactical_pitch.copy()
            elif last_tactical_pitch is not None:
                tactical_pitch = last_tactical_pitch
            else:
                tactical_pitch = None

            if tactical_pitch is not None:
                y_offset = (combined_h - tactical_h) // 2
                tactical_combined[y_offset:y_offset + tactical_h, width:width + tactical_w] = tactical_pitch

            cv2.putText(tactical_combined, "BROADCAST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(tactical_combined, "TACTICAL ANALYSIS", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            out_tactical.write(tactical_combined)

            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} frames, eval={result.get('eval_bar', 0.0):+.1f}")

            # Pass-line persistence cache (strict holder only): hold for ~0.5s and fade out.
            if current_pass_cache is not None:
                pass_cache = current_pass_cache
            if pass_cache is not None and int(pass_cache.get("remaining", 0)) > 0:
                pass_cache["remaining"] = int(pass_cache.get("remaining", 0)) - 1
                if int(pass_cache.get("remaining", 0)) <= 0:
                    pass_cache = None

            frame_idx += 1

        cap.release()
        out_eval.release()
        out_pass.release()
        out_tactical.release()

        print(f"\nDone! Generated 3 videos in {output_dir}/")
        print("  eval_overlay.mp4 - Clean broadcast with eval bar")
        print("  pass_prediction.mp4 - Broadcast with pass risk lanes")
        print("  tactical_view.mp4 - Side-by-side with Voronoi polygons + xT")
        return analyzer.eval_history

    ctx["draw_pass_lanes_on_pitch"] = _draw_pass_lanes_on_pitch_patched
    ctx["process_video_multi_output"] = _process_video_multi_output_patched

    if not getattr(SoccerAnalyzer, "_runtime_overrides_applied", False):
        original_process_frame = SoccerAnalyzer.process_frame

        def _process_frame_patched(self: Any, frame: Any):
            self._rt_frame_idx = int(getattr(self, "_rt_frame_idx", 0)) + 1
            result = original_process_frame(self, frame)
            if not isinstance(result, dict):
                return result
            if bool(result.get("analytics_paused", False)):
                return result
            _relabel_teams(self, frame, result)
            _strict_holder_from_detected_ball(self, result)
            det = result.get("detections")
            pitch_coords = result.get("pitch_coords")
            teams = result.get("teams")
            if det is not None and pitch_coords is not None and teams is not None and len(det) > 0:
                class_ids = np.asarray(det.class_id)
                result["_rt_attack_left_to_right"] = _infer_attack_left_to_right(
                    self,
                    np.asarray(pitch_coords, dtype=np.float32),
                    np.asarray(teams, dtype=int),
                    class_ids,
                    int(result.get("possession_team", -1)),
                )
            else:
                result["_rt_attack_left_to_right"] = bool(getattr(self, "_rt_attack_left_to_right_prev", True))
            _apply_eval_xt_orientation(self, result)

            return result

        SoccerAnalyzer.process_frame = _process_frame_patched
        SoccerAnalyzer._runtime_overrides_applied = True

    return ctx
