"""Bayesian Optimization (GP + Expected Improvement) to fine-tune RandomForestClassifier.

Extracts the 16 map-matching features from train/ samples, runs 5-fold
sample-grouped cross-validation guided by a GaussianProcessRegressor surrogate
with Expected Improvement, saves the tuned RF to models/rf_bayes_tuned.joblib,
and evaluates it inside Hybrid_HMM_RF on student_test and evaluator_test.
"""

import json
import math
import sys
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "hmm Model" / "src"))

from baseline_hmm import (
    load_sample,
    project_to_meters,
    smooth_track,
    build_spatial_index,
    point_to_segments,
    evaluate_route,
)
from hybrid_hmm_rf import (
    subsample_with_timestamps,
    compute_headings,
    hybrid_hmm_rf_match,
)


def extract_rf_dataset(train_dirs: list, max_pts_per_sample: int = 80):
    """Extract the 16 RF features and binary labels from train_dirs."""
    X_all, y_all, groups = [], [], []
    rng = np.random.default_rng(42)

    for g_idx, s in enumerate(train_dirs):
        nodes, arcs, track, route = load_sample(s)
        true_set = set(route)
        nodes_xy, track_xy = project_to_meters(nodes, track[:, :2])
        smoothed_xy = smooth_track(track_xy, timestamps=track[:, 2], window=7, polyorder=2)
        pts, ts = subsample_with_timestamps(smoothed_xy, track[:, 2], min_step_m=20.0)

        tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(nodes_xy, arcs)
        n_nodes = len(nodes_xy)
        in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)
        out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
        headings = compute_headings(pts)

        # Pick subset of steps for fast feature extraction
        step_indices = list(range(1, len(pts)))
        if len(step_indices) > max_pts_per_sample:
            step_indices = sorted(rng.choice(step_indices, size=max_pts_per_sample, replace=False).tolist())

        # Cache candidates for needed steps
        needed = set(step_indices) | {t - 1 for t in step_indices}
        cand_cache = {}
        for t in sorted(needed):
            p = pts[t]
            cands = np.array([], dtype=np.int32)
            for mult in (1.0, 2.5, 6.0):
                idxs = tree.query_ball_point(p, 60.0 * mult)
                if idxs:
                    cands = np.unique(all_aids[idxs])
                    break
            if len(cands) == 0:
                _, idxs = tree.query(p, k=min(8, len(all_aids)))
                cands = np.unique(all_aids[np.atleast_1d(idxs)])
            dists, fracs = point_to_segments(p, u_xy, diff, arc_lens, cands)
            order = np.argsort(dists)[:8]
            cands, dists, fracs = cands[order], dists[order], fracs[order]
            min_d = float(dists[0])
            hx, hy = headings[t]
            step_list = []
            for rank, (aid, dist, frac) in enumerate(zip(cands.tolist(), dists.tolist(), fracs.tolist())):
                L = float(arc_lens[aid])
                if L > 1e-6 and (hx != 0.0 or hy != 0.0):
                    cos_h = float(np.clip((diff[aid, 0] * hx + diff[aid, 1] * hy) / L, -1.0, 1.0))
                    h_diff = math.acos(cos_h)
                else:
                    cos_h, h_diff = 0.0, 0.0
                step_list.append((aid, dist, frac, rank, dist - min_d, dist / max(min_d, 1.0), h_diff, cos_h, L))
            cand_cache[t] = step_list

        for t in step_indices:
            d_euclid = math.hypot(pts[t, 0] - pts[t - 1, 0], pts[t, 1] - pts[t - 1, 1])
            dt = max(float(ts[t] - ts[t - 1]), 1e-3)
            speed_mps = d_euclid / dt
            # Use best previous candidate (or closest true-route prev candidate)
            prev_cands = cand_cache[t - 1]
            a_i = prev_cands[0][0]
            for c_prev in prev_cands:
                if c_prev[0] in true_set:
                    a_i = c_prev[0]
                    break
            v_i = int(arcs[a_i, 1])

            for c_j in cand_cache[t]:
                a_j, dist_j, f_j, rank_j, d_diff_j, d_rat_j, h_diff_j, cos_h_j, L_j = c_j
                u_j, v_j = int(arcs[a_j, 0]), int(arcs[a_j, 1])
                gap_dist = 0.0 if (a_i == a_j or v_i == u_j) else float(
                    math.hypot(nodes_xy[u_j, 0] - nodes_xy[v_i, 0], nodes_xy[u_j, 1] - nodes_xy[v_i, 1])
                )
                X_all.append([
                    dist_j, float(rank_j), d_diff_j, d_rat_j, h_diff_j, cos_h_j,
                    speed_mps, d_euclid, dt, L_j, f_j,
                    float(in_deg[u_j]), float(out_deg[v_j]),
                    1.0 if a_i == a_j else 0.0, 1.0 if v_i == u_j else 0.0, gap_dist,
                ])
                y_all.append(1 if a_j in true_set else 0)
                groups.append(g_idx)

    return np.asarray(X_all, dtype=np.float32), np.asarray(y_all, dtype=np.int32), np.asarray(groups, dtype=np.int32)


def expected_improvement(X_cand: np.ndarray, gp: GaussianProcessRegressor, best_y: float, xi: float = 0.01):
    """Compute Expected Improvement acquisition values for maximization."""
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - best_y - xi
    Z = imp / sigma
    return imp * norm.cdf(Z) + sigma * norm.pdf(Z)


def main():
    pkg_root = Path(__file__).resolve().parent.parent
    data_root = pkg_root.parent
    all_train = sorted([d for d in (data_root / "Student folder" / "train").iterdir() if d.is_dir()])
    train_dirs = [d for d in all_train if (d / f"{d.name}.arcs").stat().st_size < 2_500_000][::2][:25]

    t0 = time.perf_counter()
    print(f"Extracting 16 RF features from {len(train_dirs)} train samples...")
    X, y, groups = extract_rf_dataset(train_dirs, max_pts_per_sample=70)
    print(f"  Extracted X={X.shape}, positives={y.mean()*100:.1f}% in {time.perf_counter()-t0:.1f}s")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))

    # Bounds: [n_estimators, max_depth, min_samples_leaf, max_features]
    bounds = np.array([[25, 120], [5, 18], [5, 60], [0.3, 1.0]], dtype=np.float64)

    def decode_params(u):
        return {
            "n_estimators": int(round(u[0])),
            "max_depth": int(round(u[1])),
            "min_samples_leaf": int(round(u[2])),
            "max_features": float(round(u[3], 3)),
        }

    def eval_cv(u):
        p = decode_params(u)
        aucs = []
        losses = []
        for tr, va in splits:
            clf = RandomForestClassifier(
                n_estimators=p["n_estimators"],
                max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"],
                max_features=p["max_features"],
                max_samples=0.35,
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X[tr], y[tr])
            probs = clf.predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y[va], probs))
            losses.append(log_loss(y[va], probs))
        return float(np.mean(aucs)), float(np.mean(losses)), p

    # Bayesian Optimization: 5 random initial points + 10 GP-EI iterations
    rng = np.random.default_rng(42)
    X_obs, y_obs, history = [], [], []

    print("\nRunning Bayesian Optimization (5 init + 10 GP-EI iterations)...")
    for it in range(15):
        if it < 5:
            u = rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            # Normalize inputs to [0, 1]^4 for GP stability
            X_norm = (np.asarray(X_obs) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-5, normalize_y=True, random_state=42)
            gp.fit(X_norm, np.asarray(y_obs))
            cand_norm = rng.uniform(0.0, 1.0, size=(512, 4))
            ei = expected_improvement(cand_norm, gp, best_y=max(y_obs))
            best_norm = cand_norm[int(np.argmax(ei))]
            u = bounds[:, 0] + best_norm * (bounds[:, 1] - bounds[:, 0])

        auc, ll, p = eval_cv(u)
        X_obs.append(u)
        y_obs.append(auc)
        history.append((auc, ll, p))
        print(f"  Iter {it+1:>2d}/15 | AUC={auc:.4f} | LogLoss={ll:.4f} | {p}")

    history.sort(key=lambda x: x[0], reverse=True)
    best_auc, best_ll, best_p = history[0]
    print(f"\nBest Bayesian RF params: {best_p} (CV AUC={best_auc:.4f}, LogLoss={best_ll:.4f})")

    # Fit final tuned RF on all extracted train data
    final_rf = RandomForestClassifier(
        n_estimators=best_p["n_estimators"],
        max_depth=best_p["max_depth"],
        min_samples_leaf=best_p["min_samples_leaf"],
        max_features=best_p["max_features"],
        n_jobs=-1,
        random_state=42,
    )
    final_rf.fit(X, y)

    features = [
        "perp_dist", "dist_rank", "dist_diff_min", "dist_ratio", "heading_diff",
        "cos_heading_diff", "speed_mps", "step_dist", "dt", "arc_length",
        "proj_pos_ratio", "u_in_deg", "v_out_deg", "is_same_arc", "is_direct_conn", "conn_gap_dist",
    ]
    rf_out = pkg_root / "models" / "rf_bayes_tuned.joblib"
    joblib.dump({"model": final_rf, "features": features, "best_params": best_p, "cv_auc": best_auc}, rf_out)
    print(f"Saved tuned RF to {rf_out}")

    # Evaluate inside Hybrid_HMM_RF on student_test and evaluator_test
    hmm_cfg = json.loads((pkg_root / "models" / "hmm_config.json").read_text(encoding="utf-8"))
    student_samples = sorted([d for d in (data_root / "Student folder" / "student_test").iterdir() if d.is_dir()])
    eval_dir = data_root / "evaluator_test" / "evaluator_test"
    if not eval_dir.exists():
        eval_dir = data_root / "evaluator_test"
    eval_samples = sorted([d for d in eval_dir.iterdir() if d.is_dir()])

    for split_name, samples in [("student_test", student_samples), ("evaluator_test", eval_samples)]:
        cnts, lens = [], []
        tp = fp = fn = tn = exact = 0
        for s in samples:
            n, a, t, r = load_sample(s)
            xy_n, xy_t = project_to_meters(n, t[:, :2])
            pred, alens = hybrid_hmm_rf_match(xy_n, a, xy_t, t[:, 2], final_rf, hmm_cfg, rf_weight=0.5)
            m = evaluate_route(pred, r, alens)
            cnts.append(m["f1_cnt"] * 100)
            lens.append(m["f1_len"] * 100)
            ps, ts = set(pred), set(r)
            s_tp = len(ps & ts); s_fp = len(ps - ts); s_fn = len(ts - ps)
            tp += s_tp; fp += s_fp; fn += s_fn; tn += len(a) - (s_tp + s_fp + s_fn)
            if pred == r:
                exact += 1
        top1 = (tp + tn) / (tp + tn + fp + fn) * 100.0
        print(f"Hybrid (Bayes-RF, w=0.5) on {split_name}: Cnt-F1={np.mean(cnts):.2f}% | Len-F1={np.mean(lens):.2f}% | Top-1={top1:.4f}% | Exact={exact}/{len(samples)}")


if __name__ == "__main__":
    main()
