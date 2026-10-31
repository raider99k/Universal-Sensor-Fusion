from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:
    np = None  # type: ignore
    linear_sum_assignment = None  # type: ignore
    _HAS_SCIPY = False


TRACK_TENTATIVE = 0
TRACK_CONFIRMED = 1
TRACK_LOST = 2


@dataclass
class EgoMotionStep:
    """Vehicle ego-motion over the step interval [last_stamp, stamp] in base_link."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    dt: Optional[float] = None
    dx: Optional[float] = None
    dy: Optional[float] = None
    dyaw: Optional[float] = None


@dataclass
class TrackerConfig:
    mahalanobis_gate: float = 9.21
    semantic_min_overlap: float = 0.15
    semantic_cost_weight: float = 2.0
    process_noise_accel: float = 2.0
    min_meas_var: float = 0.05
    confirm_hits: int = 3
    max_missed: int = 5
    class_update_alpha: float = 0.35
    max_predict_dt: float = 0.25
    # Kinematic and belief model extensions:
    motion_model: str = "landmark"  # "landmark" (SE(2) static landmark) or "cv" (constant velocity)
    assignment_algorithm: str = "hungarian"  # "hungarian" (optimal global) or "greedy"
    semantic_model: str = "dirichlet"  # "dirichlet" (epistemic Bayesian) or "ema"
    dirichlet_prior_strength: float = 1.0
    dirichlet_evidence_scale: float = 4.0
    dirichlet_max_concentration: float = 40.0
    landmark_process_noise_pos: float = 0.03
    odom_vx_std: float = 0.15
    odom_vy_std: float = 0.15
    odom_wz_std: float = 0.05
    corridor_prior_weight: float = 0.0
    enable_global_map: bool = False
    global_map_gate_radius_m: float = 1.5
    global_map_min_lap_distance_m: float = 40.0


@dataclass
class Observation2D:
    x: float
    y: float
    z: float
    cov_xx: float
    cov_xy: float
    cov_yy: float
    z_var: float
    class_probs: List[float]
    quality: float


@dataclass
class TrackEstimate:
    track_id: int
    state: List[float]  # [x, y, vx, vy]
    covariance: List[List[float]]  # 4x4
    z: float
    z_var: float
    class_probs: List[float]
    hits: int
    misses: int
    quality: float
    status: int
    first_stamp: float
    last_stamp: float
    dirichlet_alpha: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    semantic_certainty: float = 1.0

    def clone(self) -> "TrackEstimate":
        return TrackEstimate(
            track_id=self.track_id,
            state=list(self.state),
            covariance=[list(row) for row in self.covariance],
            z=self.z,
            z_var=self.z_var,
            class_probs=list(self.class_probs),
            hits=self.hits,
            misses=self.misses,
            quality=self.quality,
            status=self.status,
            first_stamp=self.first_stamp,
            last_stamp=self.last_stamp,
            dirichlet_alpha=list(self.dirichlet_alpha),
            semantic_certainty=self.semantic_certainty,
        )


@dataclass
class TrackerSnapshot:
    tracks: Dict[int, TrackEstimate]
    next_track_id: int
    last_step_stamp: float


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_probs(probabilities: List[float]) -> List[float]:
    if not probabilities:
        return [0.25, 0.25, 0.25, 0.25]
    n = len(probabilities)
    p0 = float(probabilities[0]) if n > 0 and probabilities[0] > 0.0 else 0.0
    p1 = float(probabilities[1]) if n > 1 and probabilities[1] > 0.0 else 0.0
    p2 = float(probabilities[2]) if n > 2 and probabilities[2] > 0.0 else 0.0
    p3 = float(probabilities[3]) if n > 3 and probabilities[3] > 0.0 else 0.0
    total = p0 + p1 + p2 + p3
    if total <= 1e-9:
        return [0.25, 0.25, 0.25, 0.25]
    if abs(total - 1.0) <= 1e-6:
        return [p0, p1, p2, p3]
    inv = 1.0 / total
    return [p0 * inv, p1 * inv, p2 * inv, p3 * inv]


def _normalized_information(probs: List[float]) -> float:
    """
    Returns normalized Shannon negentropy in [0.0, 1.0].
    Uniform [0.25, 0.25, 0.25, 0.25] -> 0.0 (uninformative observation).
    One-hot [1.0, 0.0, 0.0, 0.0] -> 1.0 (maximally informative).
    """
    p = _normalize_probs(probs)
    entropy = 0.0
    for v in p:
        if v > 1e-12:
            entropy -= v * math.log(v)
    max_entropy = math.log(4.0)
    return _clamp(1.0 - (entropy / max_entropy), 0.0, 1.0)


def _se2_integrate_ego_displacement(
    vx: float, vy: float, wz: float, dt: float
) -> Tuple[float, float, float]:
    """Exact SE(2) exponential map integration of body twist (vx, vy, wz) over dt."""
    dtheta = wz * dt
    if abs(dtheta) > 1e-5:
        sin_dt = math.sin(dtheta)
        cos_dt = math.cos(dtheta)
        dx = (sin_dt * vx - (1.0 - cos_dt) * vy) / wz
        dy = ((1.0 - cos_dt) * vx + sin_dt * vy) / wz
    else:
        dx = vx * dt - 0.5 * vy * dtheta * dt
        dy = vy * dt + 0.5 * vx * dtheta * dt
    return dx, dy, dtheta


def transform_point_between_frames(
    x: float, y: float, steps: List[EgoMotionStep]
) -> Tuple[float, float]:
    """Transform a static 2D landmark (x, y) across a sequence of SE(2) ego-motion steps."""
    px, py = float(x), float(y)
    for s in steps:
        dt = float(s.dt) if s.dt is not None else 0.0
        if s.dx is not None and s.dy is not None and s.dyaw is not None:
            dx, dy, dtheta = float(s.dx), float(s.dy), float(s.dyaw)
        else:
            dx, dy, dtheta = _se2_integrate_ego_displacement(
                float(s.vx), float(s.vy), float(s.wz), dt
            )
        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)
        xt = px - dx
        yt = py - dy
        px = cos_t * xt + sin_t * yt
        py = -sin_t * xt + cos_t * yt
    return px, py


@dataclass
class CorridorEstimate:
    """Quadratic clothoid track corridor estimate in ego vehicle frame:
    y_center(x) = y0 + slope * (x - xm) + 0.5 * curvature_kappa * (x - xm)^2
    """

    xm: float
    y0: float
    slope: float
    curvature_kappa: float
    track_width_m: float
    confidence: float
    left_boundary_xy: List[Tuple[float, float]]
    right_boundary_xy: List[Tuple[float, float]]
    centerline_waypoints: List[Tuple[float, float, float, float]]


class MultiTargetEKFTracker:
    def __init__(self, config: TrackerConfig) -> None:
        self.cfg = config
        self.tracks: Dict[int, TrackEstimate] = {}
        self.next_track_id = 1
        self.last_step_stamp = -1.0
        self.last_corridor_estimate: Optional[CorridorEstimate] = None
        self.global_mapper: Optional[GlobalLandmarkMapper] = (
            GlobalLandmarkMapper(
                gate_radius_m=float(getattr(config, "global_map_gate_radius_m", 1.5)),
                min_lap_distance_m=float(
                    getattr(config, "global_map_min_lap_distance_m", 40.0)
                ),
            )
            if bool(getattr(config, "enable_global_map", False))
            else None
        )

    def snapshot(self) -> TrackerSnapshot:
        return TrackerSnapshot(
            tracks={tid: tr.clone() for tid, tr in self.tracks.items()},
            next_track_id=int(self.next_track_id),
            last_step_stamp=float(self.last_step_stamp),
        )

    def restore(self, snapshot: TrackerSnapshot) -> None:
        self.tracks = {tid: tr.clone() for tid, tr in snapshot.tracks.items()}
        self.next_track_id = int(snapshot.next_track_id)
        self.last_step_stamp = float(snapshot.last_step_stamp)

    def reset(self) -> None:
        self.tracks = {}
        self.next_track_id = 1
        self.last_step_stamp = -1.0
        if self.global_mapper is not None:
            self.global_mapper.reset()

    def get_global_landmarks(self) -> List["GlobalLandmark"]:
        if self.global_mapper is None:
            return []
        return list(self.global_mapper.landmarks.values())

    def empty_snapshot(self) -> TrackerSnapshot:
        return TrackerSnapshot(tracks={}, next_track_id=1, last_step_stamp=-1.0)

    def step(
        self,
        observations: List[Observation2D],
        stamp: float,
        ego_motion: Optional[EgoMotionStep] = None,
    ) -> Tuple[List[TrackEstimate], List[TrackEstimate], Dict[str, int]]:
        stats = {
            "tracks_created": 0,
            "tracks_deleted": 0,
            "assoc_candidates": 0,
            "assoc_matches": 0,
            "assoc_rejects": 0,
        }
        if stamp < self.last_step_stamp:
            # Keep deterministic behavior for live path; replay should not update this tracker.
            return list(self.tracks.values()), [], stats
        prev_step_stamp = self.last_step_stamp
        self.last_step_stamp = stamp

        for track in self.tracks.values():
            self._predict(
                track,
                stamp,
                ego_motion=ego_motion,
                prev_step_stamp=prev_step_stamp,
            )

        if float(getattr(self.cfg, "corridor_prior_weight", 0.0)) > 0.0:
            observations = self._regularize_observations_with_corridor_prior(
                observations, ego_motion=ego_motion
            )

        candidates: List[Tuple[float, int, int, float]] = []
        track_ids = list(self.tracks.keys())
        if track_ids and observations:
            min_var = self.cfg.min_meas_var
            gate = self.cfg.mahalanobis_gate
            min_sem = self.cfg.semantic_min_overlap
            sem_w = self.cfg.semantic_cost_weight
            obs_norm = [_normalize_probs(o.class_probs) for o in observations]
            obs_r00 = [max(min_var, o.cov_xx) for o in observations]
            obs_r11 = [max(min_var, o.cov_yy) for o in observations]
            for track_id in track_ids:
                track = self.tracks[track_id]
                tx = track.state[0]
                ty = track.state[1]
                p = track.covariance
                p00 = p[0][0]
                p01 = p[0][1]
                p10 = p[1][0]
                p11 = p[1][1]
                tp = _normalize_probs(track.class_probs)
                tp0, tp1, tp2, tp3 = tp[0], tp[1], tp[2], tp[3]
                for obs_idx, obs in enumerate(observations):
                    y0 = obs.x - tx
                    y1 = obs.y - ty
                    s00 = p00 + obs_r00[obs_idx]
                    s11 = p11 + obs_r11[obs_idx]
                    s01 = p01 + obs.cov_xy
                    s10 = p10 + obs.cov_xy
                    det = s00 * s11 - s01 * s10
                    if abs(det) < 1e-9:
                        continue
                    d2 = (y0 * (s11 * y0 - s01 * y1) + y1 * (-s10 * y0 + s00 * y1)) / det
                    if d2 < 0.0:
                        d2 = 0.0
                    if d2 <= gate:
                        op = obs_norm[obs_idx]
                        semantic_overlap = (
                            (tp0 if tp0 < op[0] else op[0])
                            + (tp1 if tp1 < op[1] else op[1])
                            + (tp2 if tp2 < op[2] else op[2])
                            + (tp3 if tp3 < op[3] else op[3])
                        )
                        if semantic_overlap >= min_sem:
                            cost = d2 + sem_w * (1.0 - semantic_overlap)
                            candidates.append((cost, track_id, obs_idx, d2))
                            stats["assoc_candidates"] += 1

        assignments = self._solve_assignments(track_ids, len(observations), candidates)
        assigned_tracks = {tid for tid, _, _ in assignments}
        assigned_obs = {oid for _, oid, _ in assignments}
        stats["assoc_matches"] = len(assignments)

        for track_id, obs_idx, d2 in assignments:
            self._update(
                self.tracks[track_id],
                observations[obs_idx],
                stamp,
                d2,
                ego_motion=ego_motion,
            )

        lost_tracks: List[TrackEstimate] = []
        for track_id in list(self.tracks.keys()):
            if track_id in assigned_tracks:
                continue
            track = self.tracks[track_id]
            track.misses += 1
            track.quality = _clamp(track.quality * 0.92, 0.0, 1.0)
            if track.misses > self.cfg.max_missed:
                track.status = TRACK_LOST
                lost_tracks.append(track)
                del self.tracks[track_id]
                stats["tracks_deleted"] += 1
            else:
                stats["assoc_rejects"] += 1

        for obs_idx, obs in enumerate(observations):
            if obs_idx in assigned_obs:
                continue
            track = self._create_track(obs, stamp, ego_motion=ego_motion)
            self.tracks[track.track_id] = track
            stats["tracks_created"] += 1

        if self.global_mapper is not None:
            dt_step = (
                max(0.0, stamp - prev_step_stamp)
                if prev_step_stamp >= 0.0
                else 0.0
            )
            self.global_mapper.step(
                tracks=list(self.tracks.values()),
                stamp=stamp,
                ego_motion=ego_motion,
                dt=dt_step,
            )

        return list(self.tracks.values()), lost_tracks, stats

    def _regularize_observations_with_corridor_prior(
        self,
        observations: List[Observation2D],
        ego_motion: Optional[EgoMotionStep] = None,
    ) -> List[Observation2D]:
        w_prior = _clamp(float(getattr(self.cfg, "corridor_prior_weight", 0.0)), 0.0, 0.90)
        if w_prior <= 1e-6 or not observations:
            return observations

        # Estimate local clothoid track centerline y_center(x) = y0 + slope * (x - xm) + 0.5 * kappa * (x - xm)^2
        # from confirmed near/mid-field landmarks (left=Blue -> y - 1.5m, right=Yellow -> y + 1.5m)
        xs_c: List[float] = []
        ys_c: List[float] = []
        n_blue_c = 0
        n_yellow_c = 0
        for tr in self.tracks.values():
            if tr.misses <= 2 and 1.8 <= tr.state[0] <= 15.5:
                tp = _normalize_probs(tr.class_probs)
                if tp[0] > 0.55 and tp[2] < 0.30:
                    xs_c.append(tr.state[0])
                    ys_c.append(tr.state[1] - 1.50)
                    n_blue_c += 1
                elif tp[1] > 0.55 and tp[2] < 0.30:
                    xs_c.append(tr.state[0])
                    ys_c.append(tr.state[1] + 1.50)
                    n_yellow_c += 1
        for obs in observations:
            if 1.8 <= obs.x <= 15.0:
                op = _normalize_probs(obs.class_probs)
                if op[0] > 0.65 and op[2] < 0.30:
                    xs_c.append(obs.x)
                    ys_c.append(obs.y - 1.50)
                    n_blue_c += 1
                elif op[1] > 0.65 and op[2] < 0.30:
                    xs_c.append(obs.x)
                    ys_c.append(obs.y + 1.50)
                    n_yellow_c += 1

        xm = 6.0
        y0 = 0.0
        slope = 0.0
        kappa = 0.0
        fit_mse = 1.0
        if ego_motion is not None and abs(float(ego_motion.vx)) > 1.5:
            kappa = _clamp(float(ego_motion.wz) / max(2.5, abs(float(ego_motion.vx))), -0.06, 0.06)
        if len(xs_c) >= 2:
            n_c = len(xs_c)
            xm = sum(xs_c) / n_c
            ym = _clamp(sum(ys_c) / n_c, -3.5, 3.5)
            y0 = ym
            us = [x - xm for x in xs_c]
            var_x = sum(u * u for u in us)
            if var_x > 1.2:
                cov_xy = sum(u * (y - ym) for u, y in zip(us, ys_c))
                slope = _clamp(cov_xy / var_x, -0.45, 0.45)
                if n_c >= 4 and (max(xs_c) - min(xs_c)) >= 4.5:
                    u2_mean = var_x / n_c
                    vs = [u * u - u2_mean for u in us]
                    var_v = sum(v * v for v in vs)
                    if var_v > 12.0:
                        cov_vy = sum(v * (y - ym - slope * u) for u, v, y in zip(us, vs, ys_c))
                        kappa_fit = _clamp(2.0 * (cov_vy / var_v), -0.06, 0.06)
                        kappa = 0.75 * kappa_fit + 0.25 * kappa if abs(kappa) > 1e-4 else kappa_fit
                        y0 = _clamp(ym - 0.5 * kappa * u2_mean, -3.5, 3.5)
                fit_mse = sum(
                    (y - (y0 + slope * u + 0.5 * kappa * u * u)) ** 2
                    for u, y in zip(us, ys_c)
                ) / n_c

        well_conditioned_corridor = (
            n_blue_c >= 2 and n_yellow_c >= 2 and len(xs_c) >= 6 and fit_mse <= 0.06
        )
        conf_corridor = _clamp(
            (min(1.0, len(xs_c) / 6.0) * math.exp(-3.0 * max(0.0, fit_mse))),
            0.0,
            1.0,
        )
        self.last_corridor_estimate = self._build_corridor_estimate(
            xm=xm,
            y0=y0,
            slope=slope,
            kappa=kappa,
            confidence=conf_corridor,
        )

        regularized: List[Observation2D] = []
        for obs in observations:
            p = _normalize_probs(obs.class_probs)
            if p[2] >= 0.35:
                regularized.append(obs)
                continue
            dx_m = obs.x - xm
            y_center = y0 + slope * dx_m + 0.5 * kappa * dx_m * dx_m
            dy = obs.y - y_center
            if abs(dy) < 0.45:
                regularized.append(obs)
                continue
            p_blue_corr = 1.0 / (1.0 + math.exp(-_clamp(2.2 * dy, -12.0, 12.0)))
            p_yellow_corr = 1.0 - p_blue_corr
            r = math.hypot(obs.x, obs.y)
            info = _normalized_information(p)
            contradicts_far_corridor = (
                well_conditioned_corridor
                and r >= 14.0
                and ((dy > 1.20 and p[1] > p[0]) or (dy < -1.20 and p[0] > p[1]))
            )
            if (0.03 <= info <= 0.45 and r >= 8.5) or contradicts_far_corridor:
                w_eff = (
                    _clamp(max(w_prior, 0.72) * min(1.0, (r - 10.0) / 5.0), 0.0, 0.85)
                    if contradicts_far_corridor
                    else _clamp(w_prior * min(1.0, (r - 7.0) / 8.0), 0.0, 0.80)
                )
                by_mass = max(0.50, p[0] + p[1])
                new_probs = _normalize_probs(
                    [
                        (1.0 - w_eff) * p[0] + w_eff * by_mass * p_blue_corr,
                        (1.0 - w_eff) * p[1] + w_eff * by_mass * p_yellow_corr,
                        p[2],
                        p[3],
                    ]
                )
            else:
                regularized.append(obs)
                continue

            regularized.append(
                Observation2D(
                    x=obs.x,
                    y=obs.y,
                    z=obs.z,
                    cov_xx=obs.cov_xx,
                    cov_xy=obs.cov_xy,
                    cov_yy=obs.cov_yy,
                    z_var=obs.z_var,
                    class_probs=new_probs,
                    quality=obs.quality,
                )
            )
        return regularized

    def _build_corridor_estimate(
        self,
        xm: float,
        y0: float,
        slope: float,
        kappa: float,
        confidence: float,
        horizon_m: float = 20.0,
        step_m: float = 1.0,
    ) -> CorridorEstimate:
        # Estimate dynamic half-width from confirmed blue/yellow track residuals around centerline
        left_offsets: List[float] = []
        right_offsets: List[float] = []
        for tr in self.tracks.values():
            if tr.misses <= 2 and 1.5 <= tr.state[0] <= horizon_m:
                tp = _normalize_probs(tr.class_probs)
                dx = tr.state[0] - xm
                yc = y0 + slope * dx + 0.5 * kappa * dx * dx
                dy = tr.state[1] - yc
                if tp[0] > 0.50 and dy > 0.5:
                    left_offsets.append(dy)
                elif tp[1] > 0.50 and dy < -0.5:
                    right_offsets.append(-dy)

        half_w_left = (
            _clamp(sum(left_offsets) / len(left_offsets), 1.25, 2.50)
            if left_offsets
            else 1.50
        )
        half_w_right = (
            _clamp(sum(right_offsets) / len(right_offsets), 1.25, 2.50)
            if right_offsets
            else 1.50
        )
        track_width_m = float(half_w_left + half_w_right)

        n_steps = max(2, int(round(horizon_m / max(0.25, step_m))) + 1)
        left_xy: List[Tuple[float, float]] = []
        right_xy: List[Tuple[float, float]] = []
        waypoints: List[Tuple[float, float, float, float]] = []
        for i in range(n_steps):
            x = min(horizon_m, i * max(0.25, step_m))
            dx = x - xm
            yc = y0 + slope * dx + 0.5 * kappa * dx * dx
            dy_dx = slope + kappa * dx
            heading = math.atan(dy_dx)
            # Normal offset (-sin(heading), cos(heading))
            nx = -math.sin(heading)
            ny = math.cos(heading)
            left_xy.append((round(x + half_w_left * nx, 4), round(yc + half_w_left * ny, 4)))
            right_xy.append((round(x - half_w_right * nx, 4), round(yc - half_w_right * ny, 4)))
            waypoints.append((round(x, 4), round(yc, 4), round(heading, 5), round(kappa, 6)))

        return CorridorEstimate(
            xm=round(float(xm), 4),
            y0=round(float(y0), 4),
            slope=round(float(slope), 5),
            curvature_kappa=round(float(kappa), 6),
            track_width_m=round(track_width_m, 4),
            confidence=round(float(confidence), 4),
            left_boundary_xy=left_xy,
            right_boundary_xy=right_xy,
            centerline_waypoints=waypoints,
        )

    def get_corridor_estimate(
        self,
        horizon_m: float = 20.0,
        step_m: float = 1.0,
        ego_motion: Optional[EgoMotionStep] = None,
    ) -> CorridorEstimate:
        """Compute or return the current quadratic clothoid track corridor and centerline waypoints."""
        if (
            self.last_corridor_estimate is not None
            and abs(horizon_m - 20.0) < 1e-3
            and abs(step_m - 1.0) < 1e-3
        ):
            return self.last_corridor_estimate

        xs_c: List[float] = []
        ys_c: List[float] = []
        for tr in self.tracks.values():
            if tr.misses <= 2 and 1.5 <= tr.state[0] <= 16.0:
                tp = _normalize_probs(tr.class_probs)
                if tp[0] > 0.50 and tp[2] < 0.35:
                    xs_c.append(tr.state[0])
                    ys_c.append(tr.state[1] - 1.50)
                elif tp[1] > 0.50 and tp[2] < 0.35:
                    xs_c.append(tr.state[0])
                    ys_c.append(tr.state[1] + 1.50)

        xm = 6.0
        y0 = 0.0
        slope = 0.0
        kappa = 0.0
        fit_mse = 1.0
        if ego_motion is not None and abs(float(ego_motion.vx)) > 1.5:
            kappa = _clamp(float(ego_motion.wz) / max(2.5, abs(float(ego_motion.vx))), -0.06, 0.06)
        if len(xs_c) >= 2:
            n_c = len(xs_c)
            xm = sum(xs_c) / n_c
            ym = _clamp(sum(ys_c) / n_c, -3.5, 3.5)
            y0 = ym
            us = [x - xm for x in xs_c]
            var_x = sum(u * u for u in us)
            if var_x > 1.2:
                cov_xy = sum(u * (y - ym) for u, y in zip(us, ys_c))
                slope = _clamp(cov_xy / var_x, -0.45, 0.45)
                if n_c >= 4 and (max(xs_c) - min(xs_c)) >= 4.0:
                    u2_mean = var_x / n_c
                    vs = [u * u - u2_mean for u in us]
                    var_v = sum(v * v for v in vs)
                    if var_v > 10.0:
                        cov_vy = sum(v * (y - ym - slope * u) for u, v, y in zip(us, vs, ys_c))
                        kappa_fit = _clamp(2.0 * (cov_vy / var_v), -0.06, 0.06)
                        kappa = 0.75 * kappa_fit + 0.25 * kappa if abs(kappa) > 1e-4 else kappa_fit
                        y0 = _clamp(ym - 0.5 * kappa * u2_mean, -3.5, 3.5)
                fit_mse = sum(
                    (y - (y0 + slope * u + 0.5 * kappa * u * u)) ** 2
                    for u, y in zip(us, ys_c)
                ) / n_c

        conf = _clamp(min(1.0, len(xs_c) / 6.0) * math.exp(-3.0 * max(0.0, fit_mse)), 0.0, 1.0)
        est = self._build_corridor_estimate(
            xm=xm,
            y0=y0,
            slope=slope,
            kappa=kappa,
            confidence=conf,
            horizon_m=horizon_m,
            step_m=step_m,
        )
        self.last_corridor_estimate = est
        return est

    def _solve_assignments(
        self,
        track_ids: List[int],
        num_obs: int,
        candidates: List[Tuple[float, int, int, float]],
    ) -> List[Tuple[int, int, float]]:
        if not candidates or not track_ids or num_obs <= 0:
            return []

        if (
            self.cfg.assignment_algorithm == "hungarian"
            and _HAS_SCIPY
            and len(track_ids) > 1
            and num_obs > 1
        ):
            disallowed_cost = 1e6
            tid_to_row = {tid: r for r, tid in enumerate(track_ids)}
            cost_matrix = np.full((len(track_ids), num_obs), disallowed_cost, dtype=np.float64)
            d2_lookup: Dict[Tuple[int, int], float] = {}
            for cost, tid, oid, d2 in candidates:
                r = tid_to_row[tid]
                cost_matrix[r, oid] = float(cost)
                d2_lookup[(r, oid)] = float(d2)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments: List[Tuple[int, int, float]] = []
            for r, c in zip(row_ind, col_ind):
                if (int(r), int(c)) in d2_lookup:
                    assignments.append((track_ids[int(r)], int(c), d2_lookup[(int(r), int(c))]))
            return assignments

        # Greedy fallback (exact for 1xN or Nx1, deterministic fallback otherwise)
        assigned_tracks = set()
        assigned_obs = set()
        assignments = []
        for cost, track_id, obs_idx, d2 in sorted(candidates, key=lambda v: v[0]):
            if track_id in assigned_tracks or obs_idx in assigned_obs:
                continue
            assigned_tracks.add(track_id)
            assigned_obs.add(obs_idx)
            assignments.append((track_id, obs_idx, d2))
        return assignments

    def _create_track(
        self,
        obs: Observation2D,
        stamp: float,
        ego_motion: Optional[EgoMotionStep] = None,
    ) -> TrackEstimate:
        var_x = max(self.cfg.min_meas_var, obs.cov_xx)
        var_y = max(self.cfg.min_meas_var, obs.cov_yy)
        cov_xy = obs.cov_xy
        covariance = [
            [var_x, cov_xy, 0.0, 0.0],
            [cov_xy, var_y, 0.0, 0.0],
            [0.0, 0.0, 0.25 if self.cfg.motion_model == "landmark" else 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.25 if self.cfg.motion_model == "landmark" else 2.0],
        ]
        vx_init = 0.0
        vy_init = 0.0
        if self.cfg.motion_model == "landmark" and ego_motion is not None:
            vx_init = -ego_motion.vx + ego_motion.wz * obs.y
            vy_init = -ego_motion.vy - ego_motion.wz * obs.x

        norm_probs = _normalize_probs(obs.class_probs)
        prior_base = max(0.05, 0.25 * self.cfg.dirichlet_prior_strength)
        info = _normalized_information(norm_probs)
        init_weight = self.cfg.dirichlet_evidence_scale * _clamp(obs.quality, 0.1, 1.0) * info
        dirichlet_alpha = [prior_base + init_weight * p for p in norm_probs]

        track = TrackEstimate(
            track_id=self.next_track_id,
            state=[obs.x, obs.y, vx_init, vy_init],
            covariance=covariance,
            z=obs.z,
            z_var=max(self.cfg.min_meas_var, obs.z_var),
            class_probs=norm_probs,
            hits=1,
            misses=0,
            quality=_clamp(obs.quality, 0.0, 1.0),
            status=TRACK_TENTATIVE,
            first_stamp=stamp,
            last_stamp=stamp,
            dirichlet_alpha=dirichlet_alpha,
            semantic_certainty=sum(dirichlet_alpha),
        )
        self.next_track_id += 1
        return track

    def _predict(
        self,
        track: TrackEstimate,
        stamp: float,
        ego_motion: Optional[EgoMotionStep] = None,
        prev_step_stamp: float = -1.0,
    ) -> None:
        ref_stamp = (
            max(track.last_stamp, prev_step_stamp)
            if prev_step_stamp >= 0.0
            else track.last_stamp
        )
        dt = _clamp(stamp - ref_stamp, 0.0, self.cfg.max_predict_dt)
        if dt <= 0.0:
            return

        if self.cfg.motion_model == "landmark":
            self._predict_landmark_se2(track, dt, ego_motion)
            return

        # Legacy Constant Velocity (CV) model
        x, y, vx, vy = track.state
        track.state = [x + vx * dt, y + vy * dt, vx, vy]

        p = track.covariance
        f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        sigma_a2 = self.cfg.process_noise_accel * self.cfg.process_noise_accel
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = [
            [0.25 * dt4 * sigma_a2, 0.0, 0.5 * dt3 * sigma_a2, 0.0],
            [0.0, 0.25 * dt4 * sigma_a2, 0.0, 0.5 * dt3 * sigma_a2],
            [0.5 * dt3 * sigma_a2, 0.0, dt2 * sigma_a2, 0.0],
            [0.0, 0.5 * dt3 * sigma_a2, 0.0, dt2 * sigma_a2],
        ]
        fp = _mat_mul_4x4(f, p)
        fpf_t = _mat_mul_4x4(fp, _mat_transpose_4x4(f))
        track.covariance = _mat_add_4x4(fpf_t, q)

    def _predict_landmark_se2(
        self,
        track: TrackEstimate,
        dt: float,
        ego_motion: Optional[EgoMotionStep],
    ) -> None:
        """
        Static landmark prediction under SE(2) vehicle ego-motion.
        Resolves the landmark physics paradox: cones do not move in the world frame.
        """
        x, y, _, _ = track.state
        vx_ego = 0.0
        vy_ego = 0.0
        wz_ego = 0.0

        if ego_motion is not None:
            vx_ego = float(ego_motion.vx)
            vy_ego = float(ego_motion.vy)
            wz_ego = float(ego_motion.wz)
            if (
                ego_motion.dx is not None
                and ego_motion.dy is not None
                and ego_motion.dyaw is not None
            ):
                dx, dy, dtheta = (
                    float(ego_motion.dx),
                    float(ego_motion.dy),
                    float(ego_motion.dyaw),
                )
            else:
                step_dt = float(ego_motion.dt) if ego_motion.dt is not None else dt
                step_dt = _clamp(step_dt, 0.0, self.cfg.max_predict_dt)
                dx, dy, dtheta = _se2_integrate_ego_displacement(
                    vx_ego, vy_ego, wz_ego, step_dt
                )
        else:
            dx, dy, dtheta = 0.0, 0.0, 0.0

        cos_t = math.cos(dtheta)
        sin_t = math.sin(dtheta)
        xt = x - dx
        yt = y - dy
        x_new = cos_t * xt + sin_t * yt
        y_new = -sin_t * xt + cos_t * yt

        vx_app = -vx_ego + wz_ego * y_new
        vy_app = -vy_ego - wz_ego * x_new
        track.state = [x_new, y_new, vx_app, vy_app]

        # Propagate 2x2 position covariance P_xy = R(-dtheta) P_xy R(-dtheta)^T + Q_odom + Q_pos
        p = track.covariance
        p00, p01, p10, p11 = p[0][0], p[0][1], p[1][0], p[1][1]

        # R * P * R^T where R = [[cos_t, sin_t], [-sin_t, cos_t]]
        rp00 = cos_t * p00 + sin_t * p10
        rp01 = cos_t * p01 + sin_t * p11
        rp10 = -sin_t * p00 + cos_t * p10
        rp11 = -sin_t * p01 + cos_t * p11

        rpr00 = rp00 * cos_t + rp01 * sin_t
        rpr01 = -rp00 * sin_t + rp01 * cos_t
        rpr10 = rp10 * cos_t + rp11 * sin_t
        rpr11 = -rp10 * sin_t + rp11 * cos_t

        # Odometry uncertainty contribution: lever arm [y_new, -x_new]^T * var(dtheta) + var(dx, dy)
        q_pos = (self.cfg.landmark_process_noise_pos ** 2) * dt
        if ego_motion is not None:
            var_dx = (self.cfg.odom_vx_std * dt) ** 2
            var_dy = (self.cfg.odom_vy_std * dt) ** 2
            var_dtheta = (self.cfg.odom_wz_std * dt) ** 2
            q00 = q_pos + var_dx + (y_new * y_new) * var_dtheta
            q11 = q_pos + var_dy + (x_new * x_new) * var_dtheta
            q01 = -(x_new * y_new) * var_dtheta
        else:
            # When no ego_motion is fed (e.g. odom frame or static test), allow mild kinematic drift
            q_static = max(q_pos, 0.25 * (self.cfg.process_noise_accel ** 2) * (dt ** 3))
            q00 = q_static
            q11 = q_static
            q01 = 0.0

        track.covariance = [
            [rpr00 + q00, 0.5 * (rpr01 + rpr10) + q01, 0.0, 0.0],
            [0.5 * (rpr01 + rpr10) + q01, rpr11 + q11, 0.0, 0.0],
            [0.0, 0.0, p[2][2], 0.0],
            [0.0, 0.0, 0.0, p[3][3]],
        ]

    def _gating_metrics(self, track: TrackEstimate, obs: Observation2D) -> Tuple[float, float]:
        p = track.covariance
        r00 = max(self.cfg.min_meas_var, obs.cov_xx)
        r11 = max(self.cfg.min_meas_var, obs.cov_yy)
        r01 = obs.cov_xy

        s00 = p[0][0] + r00
        s01 = p[0][1] + r01
        s10 = p[1][0] + r01
        s11 = p[1][1] + r11
        det = s00 * s11 - s01 * s10
        if abs(det) < 1e-9:
            return 1e9, 0.0

        y0 = obs.x - track.state[0]
        y1 = obs.y - track.state[1]
        d2 = (y0 * (s11 * y0 - s01 * y1) + y1 * (-s10 * y0 + s00 * y1)) / det
        tp = _normalize_probs(track.class_probs)
        op = _normalize_probs(obs.class_probs)
        semantic_overlap = (
            (tp[0] if tp[0] < op[0] else op[0])
            + (tp[1] if tp[1] < op[1] else op[1])
            + (tp[2] if tp[2] < op[2] else op[2])
            + (tp[3] if tp[3] < op[3] else op[3])
        )
        return float(max(0.0, d2)), float(_clamp(semantic_overlap, 0.0, 1.0))

    def _update(
        self,
        track: TrackEstimate,
        obs: Observation2D,
        stamp: float,
        d2: float,
        ego_motion: Optional[EgoMotionStep] = None,
    ) -> None:
        p = track.covariance
        r00 = max(self.cfg.min_meas_var, obs.cov_xx)
        r11 = max(self.cfg.min_meas_var, obs.cov_yy)
        r01 = obs.cov_xy
        r = [[r00, r01], [r01, r11]]

        s = [
            [p[0][0] + r[0][0], p[0][1] + r[0][1]],
            [p[1][0] + r[1][0], p[1][1] + r[1][1]],
        ]
        inv_s, valid = _invert_2x2(s)
        if not valid:
            return

        # K = P H^T S^-1 with H selecting x,y states.
        ph_t = [
            [p[0][0], p[0][1]],
            [p[1][0], p[1][1]],
            [p[2][0], p[2][1]],
            [p[3][0], p[3][1]],
        ]
        k = _mat_mul_4x2_2x2(ph_t, inv_s)

        y = [obs.x - track.state[0], obs.y - track.state[1]]
        delta = [
            k[0][0] * y[0] + k[0][1] * y[1],
            k[1][0] * y[0] + k[1][1] * y[1],
            k[2][0] * y[0] + k[2][1] * y[1],
            k[3][0] * y[0] + k[3][1] * y[1],
        ]
        track.state = [track.state[i] + delta[i] for i in range(4)]

        if self.cfg.motion_model == "landmark":
            if ego_motion is not None:
                track.state[2] = -ego_motion.vx + ego_motion.wz * track.state[1]
                track.state[3] = -ego_motion.vy - ego_motion.wz * track.state[0]
            else:
                track.state[2] = 0.0
                track.state[3] = 0.0

        # Joseph form covariance update for numerical positive-definite stability:
        # P = (I - K H) P (I - K H)^T + K R K^T
        kh = [
            [k[0][0], k[0][1], 0.0, 0.0],
            [k[1][0], k[1][1], 0.0, 0.0],
            [k[2][0], k[2][1], 0.0, 0.0],
            [k[3][0], k[3][1], 0.0, 0.0],
        ]
        i_minus_kh = _mat_sub_4x4(_eye_4x4(), kh)
        term1 = _mat_mul_4x4(_mat_mul_4x4(i_minus_kh, p), _mat_transpose_4x4(i_minus_kh))
        kr = _mat_mul_4x2_2x2(k, r)
        krk_t = [[0.0] * 4 for _ in range(4)]
        for row_i in range(4):
            for col_j in range(4):
                krk_t[row_i][col_j] = kr[row_i][0] * k[col_j][0] + kr[row_i][1] * k[col_j][1]
        track.covariance = _symmetrize_4x4(_mat_add_4x4(term1, krk_t))

        innovation_quality = math.exp(-0.5 * min(25.0, max(0.0, d2)))
        self._update_semantics(track, obs, innovation_quality)

        # Inverse-variance weighted altitude fusion
        obs_z_var = max(self.cfg.min_meas_var, obs.z_var)
        trk_z_var = max(self.cfg.min_meas_var, track.z_var)
        denom_z = obs_z_var + trk_z_var
        track.z = (trk_z_var * obs.z + obs_z_var * track.z) / denom_z
        track.z_var = max(self.cfg.min_meas_var, (obs_z_var * trk_z_var) / denom_z)

        track.hits += 1
        track.misses = 0
        track.last_stamp = stamp
        track.status = (
            TRACK_CONFIRMED if track.hits >= self.cfg.confirm_hits else TRACK_TENTATIVE
        )
        track.quality = _clamp(
            0.5 * track.quality + 0.5 * _clamp(obs.quality, 0.0, 1.0) * innovation_quality,
            0.0,
            1.0,
        )

    def _update_semantics(
        self, track: TrackEstimate, obs: Observation2D, innovation_quality: float
    ) -> None:
        class_obs = _normalize_probs(obs.class_probs)
        class_track = _normalize_probs(track.class_probs)

        if self.cfg.semantic_model == "dirichlet":
            info = _normalized_information(class_obs)
            # Only update Dirichlet concentrations if observation carries semantic information
            # (prevents uninformative [0.25, 0.25, 0.25, 0.25] blind-spot updates from washing out color)
            if info > 1e-3:
                evidence_w = (
                    self.cfg.dirichlet_evidence_scale
                    * _clamp(obs.quality, 0.05, 1.0)
                    * max(0.2, innovation_quality)
                    * info
                )
                if not track.dirichlet_alpha or len(track.dirichlet_alpha) != 4:
                    track.dirichlet_alpha = [max(0.1, p) for p in class_track]
                for i in range(4):
                    track.dirichlet_alpha[i] += evidence_w * class_obs[i]

                total_conc = sum(track.dirichlet_alpha)
                if total_conc > self.cfg.dirichlet_max_concentration:
                    scale = self.cfg.dirichlet_max_concentration / total_conc
                    track.dirichlet_alpha = [a * scale for a in track.dirichlet_alpha]
                    total_conc = self.cfg.dirichlet_max_concentration

                track.semantic_certainty = float(total_conc)
                track.class_probs = _normalize_probs(track.dirichlet_alpha)
            return

        # Legacy Exponential Moving Average (EMA)
        alpha = _clamp(self.cfg.class_update_alpha, 0.0, 1.0)
        track.class_probs = _normalize_probs(
            [alpha * class_obs[i] + (1.0 - alpha) * class_track[i] for i in range(4)]
        )


def _eye_4x4() -> List[List[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat_mul_4x4(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    b0, b1, b2, b3 = b[0], b[1], b[2], b[3]
    return [
        [
            ai[0] * b0[0] + ai[1] * b1[0] + ai[2] * b2[0] + ai[3] * b3[0],
            ai[0] * b0[1] + ai[1] * b1[1] + ai[2] * b2[1] + ai[3] * b3[1],
            ai[0] * b0[2] + ai[1] * b1[2] + ai[2] * b2[2] + ai[3] * b3[2],
            ai[0] * b0[3] + ai[1] * b1[3] + ai[2] * b2[3] + ai[3] * b3[3],
        ]
        for ai in a
    ]


def _mat_mul_4x2_2x2(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0] * 2 for _ in range(4)]
    for i in range(4):
        for j in range(2):
            out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j]
    return out


def _mat_add_4x4(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] + b[i][j] for j in range(4)] for i in range(4)]


def _mat_sub_4x4(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[a[i][j] - b[i][j] for j in range(4)] for i in range(4)]


def _mat_transpose_4x4(a: List[List[float]]) -> List[List[float]]:
    return [[a[j][i] for j in range(4)] for i in range(4)]


def _symmetrize_4x4(a: List[List[float]]) -> List[List[float]]:
    return [[0.5 * (a[i][j] + a[j][i]) for j in range(4)] for i in range(4)]


def _invert_2x2(m: List[List[float]]) -> Tuple[List[List[float]], bool]:
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if abs(det) < 1e-9:
        return [[0.0, 0.0], [0.0, 0.0]], False
    inv_det = 1.0 / det
    return (
        [
            [m[1][1] * inv_det, -m[0][1] * inv_det],
            [-m[1][0] * inv_det, m[0][0] * inv_det],
        ],
        True,
    )


@dataclass
class GlobalLandmark:
    """Persistent world-frame cone landmark with 2x2 position uncertainty and Dirichlet semantics."""

    landmark_id: int
    wx: float
    wy: float
    wz: float
    cov_xx: float
    cov_xy: float
    cov_yy: float
    dirichlet_alpha: List[float]
    class_probs: List[float]
    observations_count: int
    last_seen_stamp: float


class GlobalLandmarkMapper:
    """
    Lightweight, pluggable global pose-landmark map maintainer & start-gate loop-closure corrector.
    Optional module (`enable_global_map=True`) for teams that do not run an external GraphSLAM node
    or want multi-lap landmark persistence and drift correction out of the box.
    """

    def __init__(
        self,
        gate_radius_m: float = 1.5,
        min_lap_distance_m: float = 40.0,
        loop_closure_gain: float = 0.65,
    ) -> None:
        self.gate_radius_m = max(0.3, float(gate_radius_m))
        self.min_lap_distance_m = max(10.0, float(min_lap_distance_m))
        self.loop_closure_gain = _clamp(float(loop_closure_gain), 0.0, 1.0)
        self.reset()

    def reset(self) -> None:
        self.pose_x: float = 0.0
        self.pose_y: float = 0.0
        self.pose_yaw: float = 0.0
        self.total_distance_m: float = 0.0
        self.last_loop_closure_dist_m: float = 0.0
        self.loop_closures_count: int = 0
        self.next_landmark_id: int = 1
        self.landmarks: Dict[int, GlobalLandmark] = {}

    def step(
        self,
        tracks: List[TrackEstimate],
        stamp: float,
        ego_motion: Optional[EgoMotionStep] = None,
        dt: float = 0.0,
    ) -> None:
        if ego_motion is not None:
            if (
                ego_motion.dx is not None
                and ego_motion.dy is not None
                and ego_motion.dyaw is not None
            ):
                dx, dy, dyaw = (
                    float(ego_motion.dx),
                    float(ego_motion.dy),
                    float(ego_motion.dyaw),
                )
            else:
                step_dt = float(ego_motion.dt) if ego_motion.dt is not None else dt
                dx, dy, dyaw = _se2_integrate_ego_displacement(
                    float(ego_motion.vx),
                    float(ego_motion.vy),
                    float(ego_motion.wz),
                    max(0.0, step_dt),
                )
            c_y = math.cos(self.pose_yaw)
            s_y = math.sin(self.pose_yaw)
            self.pose_x += c_y * dx - s_y * dy
            self.pose_y += s_y * dx + c_y * dy
            self.pose_yaw = math.atan2(
                math.sin(self.pose_yaw + dyaw), math.cos(self.pose_yaw + dyaw)
            )
            self.total_distance_m += math.hypot(dx, dy)

        confirmed = [
            tr
            for tr in tracks
            if tr.status == TRACK_CONFIRMED and tr.misses == 0
        ]
        if not confirmed:
            return

        c_y = math.cos(self.pose_yaw)
        s_y = math.sin(self.pose_yaw)
        gate_r2 = self.gate_radius_m * self.gate_radius_m

        # Check for start-gate / multi-lap loop closure before updating landmarks
        if (
            self.total_distance_m - self.last_loop_closure_dist_m
            >= self.min_lap_distance_m
            and len(self.landmarks) >= 4
        ):
            corr_dx: List[float] = []
            corr_dy: List[float] = []
            for tr in confirmed:
                bx, by = tr.state[0], tr.state[1]
                if math.hypot(bx, by) > 15.0:
                    continue
                wx = self.pose_x + c_y * bx - s_y * by
                wy = self.pose_y + s_y * bx + c_y * by
                tp = _normalize_probs(tr.class_probs)
                best_lm: Optional[GlobalLandmark] = None
                best_d2 = max(gate_r2, 2.5 * 2.5)
                for lm in self.landmarks.values():
                    if lm.observations_count < 3 or (stamp - lm.last_seen_stamp) < 5.0:
                        continue
                    # Match either Orange start-gate cones or high-confidence established landmarks
                    sem_sim = sum(min(a, b) for a, b in zip(tp, lm.class_probs))
                    if sem_sim < 0.45:
                        continue
                    d2 = (wx - lm.wx) ** 2 + (wy - lm.wy) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best_lm = lm
                if best_lm is not None:
                    corr_dx.append(best_lm.wx - wx)
                    corr_dy.append(best_lm.wy - wy)
            if len(corr_dx) >= 2:
                mean_dx = sum(corr_dx) / len(corr_dx)
                mean_dy = sum(corr_dy) / len(corr_dy)
                if math.hypot(mean_dx, mean_dy) <= 3.0:
                    self.pose_x += self.loop_closure_gain * mean_dx
                    self.pose_y += self.loop_closure_gain * mean_dy
                    self.last_loop_closure_dist_m = self.total_distance_m
                    self.loop_closures_count += 1
                    c_y = math.cos(self.pose_yaw)
                    s_y = math.sin(self.pose_yaw)

        for tr in confirmed:
            bx, by = tr.state[0], tr.state[1]
            wx = self.pose_x + c_y * bx - s_y * by
            wy = self.pose_y + s_y * bx + c_y * by
            p00 = max(1e-4, tr.covariance[0][0])
            p01 = tr.covariance[0][1]
            p11 = max(1e-4, tr.covariance[1][1])
            w_cov_xx = c_y * (c_y * p00 - s_y * p01) - s_y * (c_y * p01 - s_y * p11)
            w_cov_yy = s_y * (s_y * p00 + c_y * p01) + c_y * (s_y * p01 + c_y * p11)
            w_cov_xy = c_y * (s_y * p00 + c_y * p01) - s_y * (s_y * p01 + c_y * p11)
            tp = _normalize_probs(tr.class_probs)

            best_id = -1
            best_d2 = gate_r2
            for lid, lm in self.landmarks.items():
                d2 = (wx - lm.wx) ** 2 + (wy - lm.wy) ** 2
                if d2 < best_d2:
                    sem_sim = sum(min(a, b) for a, b in zip(tp, lm.class_probs))
                    if sem_sim >= 0.20:
                        best_d2 = d2
                        best_id = lid

            if best_id >= 0:
                lm = self.landmarks[best_id]
                # 2D Kalman update in global frame
                s00 = lm.cov_xx + w_cov_xx
                s01 = lm.cov_xy + w_cov_xy
                s11 = lm.cov_yy + w_cov_yy
                det = s00 * s11 - s01 * s01
                if det > 1e-9:
                    inv_det = 1.0 / det
                    k00 = (lm.cov_xx * s11 - lm.cov_xy * s01) * inv_det
                    k01 = (-lm.cov_xx * s01 + lm.cov_xy * s00) * inv_det
                    k10 = (lm.cov_xy * s11 - lm.cov_yy * s01) * inv_det
                    k11 = (-lm.cov_xy * s01 + lm.cov_yy * s00) * inv_det
                    iy0 = wx - lm.wx
                    iy1 = wy - lm.wy
                    lm.wx += k00 * iy0 + k01 * iy1
                    lm.wy += k10 * iy0 + k11 * iy1
                    lm.wz = 0.5 * (lm.wz + tr.z)
                    lm.cov_xx = max(1e-4, (1.0 - k00) * lm.cov_xx - k01 * lm.cov_xy)
                    lm.cov_yy = max(1e-4, -k10 * lm.cov_xy + (1.0 - k11) * lm.cov_yy)
                    lm.cov_xy = (1.0 - k00) * lm.cov_xy - k01 * lm.cov_yy
                info = _normalized_information(tp)
                if info > 1e-3:
                    for i in range(4):
                        lm.dirichlet_alpha[i] = min(
                            60.0, lm.dirichlet_alpha[i] + 1.5 * info * tp[i]
                        )
                    lm.class_probs = _normalize_probs(lm.dirichlet_alpha)
                lm.observations_count += 1
                lm.last_seen_stamp = stamp
            else:
                lid = self.next_landmark_id
                self.next_landmark_id += 1
                self.landmarks[lid] = GlobalLandmark(
                    landmark_id=lid,
                    wx=wx,
                    wy=wy,
                    wz=tr.z,
                    cov_xx=max(1e-3, w_cov_xx),
                    cov_xy=w_cov_xy,
                    cov_yy=max(1e-3, w_cov_yy),
                    dirichlet_alpha=list(tr.dirichlet_alpha)
                    if len(tr.dirichlet_alpha) == 4
                    else [1.0 * p for p in tp],
                    class_probs=tp,
                    observations_count=1,
                    last_seen_stamp=stamp,
                )

    def get_local_landmarks(self, max_range_m: float = 25.0) -> List[Observation2D]:
        """Reprojects established global landmarks into the current vehicle `base_link` frame."""
        c_y = math.cos(self.pose_yaw)
        s_y = math.sin(self.pose_yaw)
        max_r2 = max_range_m * max_range_m
        out: List[Observation2D] = []
        for lm in self.landmarks.values():
            if lm.observations_count < 2:
                continue
            dx = lm.wx - self.pose_x
            dy = lm.wy - self.pose_y
            if (dx * dx + dy * dy) > max_r2:
                continue
            bx = c_y * dx + s_y * dy
            by = -s_y * dx + c_y * dy
            if bx < 0.0:
                continue
            out.append(
                Observation2D(
                    x=bx,
                    y=by,
                    z=lm.wz,
                    cov_xx=lm.cov_xx,
                    cov_xy=lm.cov_xy,
                    cov_yy=lm.cov_yy,
                    z_var=0.05,
                    class_probs=list(lm.class_probs),
                    quality=min(1.0, 0.5 + 0.1 * lm.observations_count),
                )
            )
        return out

