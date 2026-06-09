#!/usr/bin/env python3
"""
Baseline PW-NDT trajectory visualisation and KITTI-style evaluation.
Usage:
    python3 eval_baseline.py [--res PATH] [--gt PATH] [--out DIR]
Defaults use the standard result/poses paths for sequence
2019-01-10-12-32-52-radar-oxford-10k.
"""
import argparse, sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless — saves PNG files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SEQ   = "2019-01-10-12-32-52-radar-oxford-10k"
BASE  = "/home/mcw/Desktop/MAY16PhD/PW-ndt/PW-ndt/catkin_ws/src"
RES_DEFAULT = f"{BASE}/results/result/data/{SEQ}_res_baseline.txt"
GT_DEFAULT  = f"{BASE}/poses/{SEQ}_gt.txt"
OUT_DEFAULT = "/home/mcw/Desktop/MAY16PhD/PW-ndt/PW-ndt/eval_output"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--res", default=RES_DEFAULT)
    p.add_argument("--gt",  default=GT_DEFAULT)
    p.add_argument("--out", default=OUT_DEFAULT)
    return p.parse_args()

# ──────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────
def load_kitti_poses(path):
    """
    Read KITTI-format pose file (N lines, each 12 floats = 3×4 matrix row-major).
    Returns (N,4,4) ndarray of homogeneous poses.
    """
    poses = []
    with open(path) as f:
        for line in f:
            v = list(map(float, line.split()))
            if len(v) != 12:
                continue
            T = np.eye(4)
            T[0] = v[0:4]
            T[1] = v[4:8]
            T[2] = v[8:12]
            poses.append(T)
    return np.array(poses)

def prepend_identity(poses):
    """Prepend a 4×4 identity to a (N,4,4) array → (N+1,4,4)."""
    return np.vstack([np.eye(4)[None], poses])

def poses_to_xy(poses):
    """Extract (N,2) XY positions from (N,4,4) pose array."""
    return poses[:, :2, 3]

# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────
def ate_trans(est_poses, gt_poses):
    """Absolute Trajectory Error — translation RMSE (metres)."""
    n = min(len(est_poses), len(gt_poses))
    diffs = est_poses[:n, :3, 3] - gt_poses[:n, :3, 3]
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))

def rpe_stats(est_poses, gt_poses, delta=1):
    """
    Relative Pose Error over delta-frame windows.
    Returns mean translational RPE (m) and mean rotational RPE (deg).
    """
    n = min(len(est_poses), len(gt_poses))
    t_errs, r_errs = [], []
    for i in range(0, n - delta):
        Q_est = np.linalg.inv(est_poses[i]) @ est_poses[i + delta]
        Q_gt  = np.linalg.inv(gt_poses[i])  @ gt_poses[i + delta]
        E     = np.linalg.inv(Q_gt) @ Q_est
        t_errs.append(np.linalg.norm(E[:3, 3]))
        # rotation angle from trace
        cos_a = (np.trace(E[:3, :3]) - 1) / 2.0
        cos_a = np.clip(cos_a, -1, 1)
        r_errs.append(np.degrees(np.abs(np.arccos(cos_a))))
    return (float(np.mean(t_errs)) if t_errs else np.nan,
            float(np.mean(r_errs)) if r_errs else np.nan)

def kitti_tr_rr(est_poses, gt_poses,
                path_lengths=(100, 200, 300, 400, 500, 600, 700, 800),
                step_size=4):
    """
    KITTI translational (%) and rotational (rad/m, deg/m) errors.
    Matches the official devkit_for_oxford C++ evaluator exactly:
      - step_size=4  (every 1 second at 4 Hz Oxford radar)
      - t_err normalised by actual GT sub-sequence length (not by Q_gt norm)
      - Rr reported in rad/m (matches spreadsheet) and deg/m
    """
    n = min(len(est_poses), len(gt_poses))

    # Cumulative GT path distances (same as C++ trajectoryDistances)
    cum_dist = [0.0]
    for i in range(1, n):
        cum_dist.append(cum_dist[-1] + np.linalg.norm(gt_poses[i, :3, 3] - gt_poses[i-1, :3, 3]))
    cum_dist = np.array(cum_dist)

    tr_errors, rr_errors = [], []
    for start in range(0, n, step_size):          # step_size=4: every 4 frames = every 1 s
        for L in path_lengths:
            # Find last frame where cumulative distance from start >= L
            target = cum_dist[start] + L
            candidates = np.where(cum_dist[start:] >= target)[0]
            if len(candidates) == 0:
                continue
            end = start + candidates[0]
            if end >= n:
                continue

            Q_est = np.linalg.inv(est_poses[start]) @ est_poses[end]
            Q_gt  = np.linalg.inv(gt_poses[start])  @ gt_poses[end]
            E     = np.linalg.inv(Q_gt) @ Q_est          # error transform

            t_err = np.linalg.norm(E[:3, 3])              # metres
            cos_a = np.clip((np.trace(E[:3, :3]) - 1) / 2.0, -1, 1)
            r_err = float(np.abs(np.arccos(cos_a)))       # radians

            # Normalise by actual GT sub-sequence length (matches C++ t_err/len)
            actual_len = cum_dist[end] - cum_dist[start]
            if actual_len > 0:
                tr_errors.append(t_err / actual_len * 100.0)   # %
                rr_errors.append(r_err / actual_len)            # rad/m

    mean_tr    = float(np.mean(tr_errors)) if tr_errors else np.nan
    mean_rr_rm = float(np.mean(rr_errors)) if rr_errors else np.nan     # rad/m
    mean_rr_dm = float(np.degrees(mean_rr_rm)) if np.isfinite(mean_rr_rm) else np.nan  # deg/m
    return mean_tr, mean_rr_rm, mean_rr_dm

def final_drift(est_poses, gt_poses):
    """Euclidean distance between final estimated and GT position (metres)."""
    n = min(len(est_poses), len(gt_poses))
    return float(np.linalg.norm(
        est_poses[n-1, :3, 3] - gt_poses[n-1, :3, 3]))

def path_length(poses):
    """Total path length in metres."""
    d = 0.0
    for i in range(1, len(poses)):
        d += np.linalg.norm(poses[i, :3, 3] - poses[i-1, :3, 3])
    return d

# ──────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────
def plot_trajectory(est_xy, gt_xy, out_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(gt_xy[:, 0],  gt_xy[:, 1],  'k-',  linewidth=1.2, label='Ground Truth')
    ax.plot(est_xy[:, 0], est_xy[:, 1], 'r-',  linewidth=1.0, label='PW-NDT Baseline')
    ax.plot(gt_xy[0, 0],  gt_xy[0, 1],  'go',  markersize=8,  label='Start')
    ax.plot(gt_xy[-1, 0], gt_xy[-1, 1], 'bs',  markersize=8,  label='End (GT)')
    ax.plot(est_xy[-1, 0],est_xy[-1, 1],'r^',  markersize=8,  label='End (Est)')
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('PW-NDT Baseline — Trajectory vs Ground Truth\n'
                 f'{SEQ}', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_per_frame_error(est_poses, gt_poses, out_path):
    n = min(len(est_poses), len(gt_poses))
    trans_err = np.linalg.norm(est_poses[:n, :3, 3] - gt_poses[:n, :3, 3], axis=1)
    frames = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(frames, trans_err, 'r-', linewidth=0.6, alpha=0.8)
    axes[0].set_ylabel('Trans. Error (m)', fontsize=11)
    axes[0].set_title('Per-Frame Absolute Position Error', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Cumulative path length on x-axis for second subplot
    cum = [0.0]
    for i in range(1, n):
        cum.append(cum[-1] + np.linalg.norm(gt_poses[i, :3, 3] - gt_poses[i-1, :3, 3]))
    axes[1].plot(cum, trans_err, 'b-', linewidth=0.6, alpha=0.8)
    axes[1].set_xlabel('Cumulative GT Path (m)', fontsize=11)
    axes[1].set_ylabel('Trans. Error (m)', fontsize=11)
    axes[1].set_title('Position Error vs Distance Travelled', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

def plot_per_frame_xy_error(est_poses, gt_poses, out_path):
    n = min(len(est_poses), len(gt_poses))
    frames = np.arange(n)
    dx = est_poses[:n, 0, 3] - gt_poses[:n, 0, 3]
    dy = est_poses[:n, 1, 3] - gt_poses[:n, 1, 3]

    # Extract yaw error
    yaw_est = np.arctan2(est_poses[:n, 1, 0], est_poses[:n, 0, 0])
    yaw_gt  = np.arctan2(gt_poses[:n, 1, 0],  gt_poses[:n, 0, 0])
    dyaw = np.degrees(np.arctan2(np.sin(yaw_est - yaw_gt), np.cos(yaw_est - yaw_gt)))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(frames, dx, 'r-', lw=0.6, alpha=0.8); axes[0].set_ylabel('ΔX (m)'); axes[0].grid(alpha=0.3)
    axes[1].plot(frames, dy, 'g-', lw=0.6, alpha=0.8); axes[1].set_ylabel('ΔY (m)'); axes[1].grid(alpha=0.3)
    axes[2].plot(frames, dyaw,'b-',lw=0.6, alpha=0.8); axes[2].set_ylabel('Δθ (deg)'); axes[2].grid(alpha=0.3)
    axes[2].set_xlabel('Frame index')
    fig.suptitle('Per-Frame X / Y / Yaw Drift — PW-NDT Baseline', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PW-NDT Baseline Evaluation")
    print(f"  Result : {args.res}")
    print(f"  GT     : {args.gt}")
    print(f"{'='*60}")

    if not os.path.exists(args.res):
        print(f"[ERROR] Result file not found: {args.res}")
        print("  The run may still be in progress. Check /tmp/baseline_run.log")
        sys.exit(1)
    if not os.path.exists(args.gt):
        print(f"[ERROR] GT file not found: {args.gt}")
        sys.exit(1)

    # ── Load ──
    res_raw = load_kitti_poses(args.res)
    gt_raw  = load_kitti_poses(args.gt)
    print(f"\n  Loaded result poses : {len(res_raw)}")
    print(f"  Loaded GT poses     : {len(gt_raw)}")

    # The code prepends Identity at index 0, and GT file has cumulative poses.
    # Result file: line k = pose at frame k+1 (frame 0 initialises, doesn't write).
    # We prepend Identity to both so indices align.
    est_poses = prepend_identity(res_raw)   # index 0 = origin
    gt_poses  = prepend_identity(gt_raw)    # index 0 = identity (frame 0)

    n_common = min(len(est_poses), len(gt_poses))
    print(f"  Common frames for eval: {n_common}")

    est_poses = est_poses[:n_common]
    gt_poses  = gt_poses[:n_common]

    # ── Metrics ──
    print(f"\n{'─'*50}")
    print(f"  METRICS")
    print(f"{'─'*50}")

    total_path = path_length(gt_poses)
    print(f"  Total GT path length  : {total_path:.1f} m")

    ate = ate_trans(est_poses, gt_poses)
    print(f"  ATE (translation RMSE): {ate:.3f} m")

    drift = final_drift(est_poses, gt_poses)
    drift_pct = drift / total_path * 100
    print(f"  Final drift           : {drift:.2f} m  ({drift_pct:.2f}% of path)")

    rpe_t, rpe_r = rpe_stats(est_poses, gt_poses, delta=1)
    print(f"  RPE (δ=1) trans       : {rpe_t*100:.4f} cm/frame")
    print(f"  RPE (δ=1) rot         : {rpe_r:.4f} deg/frame")

    # KITTI metrics — only if enough path coverage
    if total_path >= 100:
        tr_pct, rr_radm, rr_degm = kitti_tr_rr(est_poses, gt_poses)
        print(f"\n  KITTI Translational Error (Tr%) : {tr_pct:.4f} %  ({tr_pct/100:.6f})")
        print(f"  KITTI Rotational Error  (Rr)    : {rr_radm:.6f} rad/m  ({rr_degm:.4f} deg/m)")
    else:
        print(f"\n  (KITTI eval skipped — path < 100 m)")
        tr_pct, rr_radm, rr_degm = np.nan, np.nan, np.nan

    print(f"{'─'*50}")

    # ── Save metrics txt ──
    metrics_path = os.path.join(args.out, "baseline_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Sequence       : {SEQ}\n")
        f.write(f"Result file    : {args.res}\n")
        f.write(f"GT file        : {args.gt}\n")
        f.write(f"Frames eval    : {n_common}\n")
        f.write(f"GT path (m)    : {total_path:.2f}\n")
        f.write(f"ATE (m)        : {ate:.4f}\n")
        f.write(f"Final drift (m): {drift:.4f}\n")
        f.write(f"Final drift (%): {drift_pct:.4f}\n")
        f.write(f"RPE-t (cm/frm) : {rpe_t*100:.4f}\n")
        f.write(f"RPE-r (deg/frm): {rpe_r:.4f}\n")
        f.write(f"KITTI Tr%        : {tr_pct:.6f}\n")
        f.write(f"KITTI Tr (frac)  : {tr_pct/100:.6f}\n")
        f.write(f"KITTI Rr (rad/m) : {rr_radm:.6f}\n")
        f.write(f"KITTI Rr (deg/m) : {rr_degm:.6f}\n")
    print(f"\n  Metrics saved: {metrics_path}")

    # ── Plots ──
    print(f"\n  Generating plots ...")
    est_xy = poses_to_xy(est_poses)
    gt_xy  = poses_to_xy(gt_poses)

    plot_trajectory(est_xy, gt_xy,
                    os.path.join(args.out, "trajectory.png"))
    plot_per_frame_error(est_poses, gt_poses,
                         os.path.join(args.out, "per_frame_error.png"))
    plot_per_frame_xy_error(est_poses, gt_poses,
                            os.path.join(args.out, "xy_yaw_drift.png"))

    print(f"\n  All outputs in: {args.out}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
