"""Compare Untuned RF vs Bayesian-Tuned RF on identical partitions & plot BO loss curve."""

import math
import time
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from baseline_hmm import load_sample, project_to_meters, evaluate_route
from bayes_tune_rf import extract_rf_dataset
from rf_viterbi import rf_viterbi_reconstruct


def eval_candidate_metrics(clf, X, y, groups):
    """Evaluate candidate-level Macro-F1, Log Loss, and Top-1 accuracy per step."""
    t0 = time.perf_counter()
    probs = clf.predict_proba(X)[:, 1]
    inf_time = time.perf_counter() - t0
    preds = (probs >= 0.5).astype(np.int32)
    macro_f1 = f1_score(y, preds, average="macro")
    ll = log_loss(y, np.clip(probs, 1e-6, 1.0 - 1e-6))
    top1 = float((preds == y).mean())
    return macro_f1, ll, top1, inf_time


def main():
    pkg_root = Path(__file__).resolve().parent.parent
    data_root = pkg_root.parent
    all_train = sorted([d for d in (data_root / "Student folder" / "train").iterdir() if d.is_dir()])
    train_dirs = [d for d in all_train if (d / f"{d.name}.arcs").stat().st_size < 2_500_000][::2][:25]
    test_dirs = sorted([d for d in (data_root / "Student folder" / "student_test").iterdir() if d.is_dir()])

    X_tr, y_tr, g_tr = extract_rf_dataset(train_dirs, max_pts_per_sample=70)
    X_te, y_te, g_te = extract_rf_dataset(test_dirs, max_pts_per_sample=200)

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_tr, y_tr, g_tr))

    # 1. Untuned RF (sklearn defaults: n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt')
    untuned_cv_f1, untuned_cv_ll = [], []
    for tr, va in splits:
        c = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        c.fit(X_tr[tr], y_tr[tr])
        p = c.predict_proba(X_tr[va])[:, 1]
        untuned_cv_f1.append(f1_score(y_tr[va], (p >= 0.5).astype(int), average="macro"))
        untuned_cv_ll.append(log_loss(y_tr[va], np.clip(p, 1e-6, 1.0 - 1e-6)))

    untuned_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    untuned_rf.fit(X_tr, y_tr)
    u_te_f1, u_te_ll, u_te_top1, u_cand_t = eval_candidate_metrics(untuned_rf, X_te, y_te, g_te)

    # 2. Bayesian Optimization (15 trials: 5 init + 10 GP-EI) tracking CV Macro-F1 and Log Loss
    bounds = np.array([[25, 120], [5, 18], [5, 60], [0.3, 1.0]], dtype=np.float64)
    rng = np.random.default_rng(42)
    X_obs, y_obs = [], []
    trial_losses, trial_f1s, trial_params = [], [], []

    for it in range(15):
        if it < 5:
            u = rng.uniform(bounds[:, 0], bounds[:, 1])
        else:
            X_norm = (np.asarray(X_obs) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-5, normalize_y=True, random_state=42)
            gp.fit(X_norm, np.asarray(y_obs))
            cand_norm = rng.uniform(0.0, 1.0, size=(512, 4))
            mu, sigma = gp.predict(cand_norm, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            imp = mu - max(y_obs) - 0.01
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            u = bounds[:, 0] + cand_norm[int(np.argmax(ei))] * (bounds[:, 1] - bounds[:, 0])

        p = {
            "n_estimators": int(round(u[0])),
            "max_depth": int(round(u[1])),
            "min_samples_leaf": int(round(u[2])),
            "max_features": float(round(u[3], 3)),
        }
        f1s, lls = [], []
        for tr, va in splits:
            c = RandomForestClassifier(**p, max_samples=0.35, n_jobs=-1, random_state=42)
            c.fit(X_tr[tr], y_tr[tr])
            pr = c.predict_proba(X_tr[va])[:, 1]
            f1s.append(f1_score(y_tr[va], (pr >= 0.5).astype(int), average="macro"))
            lls.append(log_loss(y_tr[va], np.clip(pr, 1e-6, 1.0 - 1e-6)))
        mean_f1, mean_ll = float(np.mean(f1s)), float(np.mean(lls))
        X_obs.append(u)
        y_obs.append(-mean_ll)  # minimize Log Loss
        trial_losses.append(mean_ll)
        trial_f1s.append(mean_f1)
        trial_params.append(p)

    best_idx = int(np.argmin(trial_losses))
    best_p = trial_params[best_idx]
    tuned_rf = RandomForestClassifier(**best_p, n_jobs=-1, random_state=42)
    tuned_rf.fit(X_tr, y_tr)
    t_te_f1, t_te_ll, t_te_top1, t_cand_t = eval_candidate_metrics(tuned_rf, X_te, y_te, g_te)

    # Route-level comparison: Raw RF (no Viterbi) vs RF + Viterbi
    def eval_route_mode(clf, use_viterbi: bool):
        t0 = time.perf_counter()
        cnts, lens = [], []
        for s in test_dirs:
            n, a, t, r = load_sample(s)
            xy_n, xy_t = project_to_meters(n, t[:, :2])
            if use_viterbi:
                pred, alens = rf_viterbi_reconstruct(xy_n, a, xy_t, t[:, 2], clf)
            else:
                # Raw nearest-candidate argmax without Viterbi transition decoding
                from baseline_hmm import smooth_track, build_spatial_index, point_to_segments
                from hybrid_hmm_rf import subsample_with_timestamps, compute_headings
                sm = smooth_track(xy_t, timestamps=t[:, 2], window=7, polyorder=2)
                pts, ts = subsample_with_timestamps(sm, t[:, 2], min_step_m=20.0)
                tree, all_aids, u_xy, _, diff, alens = build_spatial_index(xy_n, a)
                in_deg = np.bincount(a[:, 1], minlength=len(xy_n))
                out_deg = np.bincount(a[:, 0], minlength=len(xy_n))
                headings = compute_headings(pts)
                pred = []
                for idx_t, pt in enumerate(pts):
                    idxs = tree.query_ball_point(pt, 150.0)
                    cands = np.unique(all_aids[idxs]) if idxs else np.unique(all_aids[np.atleast_1d(tree.query(pt, k=12)[1])])
                    dists, fracs = point_to_segments(pt, u_xy, diff, alens, cands)
                    order = np.argsort(dists)[:12]
                    cands, dists, fracs = cands[order], dists[order], fracs[order]
                    min_d = float(dists[0])
                    d_euclid = math.hypot(pts[idx_t, 0] - pts[max(0, idx_t - 1), 0], pts[idx_t, 1] - pts[max(0, idx_t - 1), 1])
                    dt = max(float(ts[idx_t] - ts[max(0, idx_t - 1)]), 1e-3) if idx_t > 0 else 1.0
                    hx, hy = headings[idx_t]
                    feats = []
                    for rnk, (aid, dst, frc) in enumerate(zip(cands.tolist(), dists.tolist(), fracs.tolist())):
                        L = float(alens[aid])
                        ch = float(np.clip((diff[aid, 0] * hx + diff[aid, 1] * hy) / L, -1.0, 1.0)) if L > 1e-6 and (hx or hy) else 0.0
                        hd = math.acos(ch) if L > 1e-6 and (hx or hy) else 0.0
                        feats.append([dst, float(rnk), dst - min_d, dst / max(min_d, 1.0), hd, ch, d_euclid / dt, d_euclid, dt, L, frc, float(in_deg[a[aid, 0]]), float(out_deg[a[aid, 1]]), 0.0, 0.0, 0.0])
                    best_a = int(cands[int(np.argmax(clf.predict_proba(np.asarray(feats, dtype=np.float32))[:, 1]))])
                    if not pred or pred[-1] != best_a:
                        pred.append(best_a)
            m = evaluate_route(pred, r, alens)
            cnts.append(m["f1_cnt"] * 100)
            lens.append(m["f1_len"] * 100)
        return float(np.mean(cnts)), float(np.mean(lens)), time.perf_counter() - t0

    raw_cnt, raw_len, raw_t = eval_route_mode(tuned_rf, use_viterbi=False)
    vit_cnt, vit_len, vit_t = eval_route_mode(tuned_rf, use_viterbi=True)
    unt_vit_cnt, unt_vit_len, unt_vit_t = eval_route_mode(untuned_rf, use_viterbi=True)

    print("\n=== UNTUNED vs BAYESIAN-TUNED RF (identical partitions) ===")
    print(f"Untuned RF: CV Macro-F1={np.mean(untuned_cv_f1):.4f} | CV LogLoss={np.mean(untuned_cv_ll):.4f} | Test Top-1={u_te_top1*100:.2f}% | Test Macro-F1={u_te_f1:.4f} | Test LogLoss={u_te_ll:.4f} | Cand Inf={u_cand_t:.3f}s | Route Viterbi=({unt_vit_cnt:.2f}%/{unt_vit_len:.2f}%, {unt_vit_t:.2f}s)")
    print(f"Tuned RF  : CV Macro-F1={trial_f1s[best_idx]:.4f} | CV LogLoss={trial_losses[best_idx]:.4f} | Test Top-1={t_te_top1*100:.2f}% | Test Macro-F1={t_te_f1:.4f} | Test LogLoss={t_te_ll:.4f} | Cand Inf={t_cand_t:.3f}s | Route Viterbi=({vit_cnt:.2f}%/{vit_len:.2f}%, {vit_t:.2f}s)")
    print(f"Best params (Trial {best_idx+1}): {best_p}")
    print(f"\n=== ROUTE RECONSTRUCTION ABLATION (Tuned RF) ===")
    print(f"Raw RF (no Viterbi): Cnt-F1={raw_cnt:.2f}% | Len-F1={raw_len:.2f}% | Time={raw_t:.2f}s")
    print(f"RF + Viterbi       : Cnt-F1={vit_cnt:.2f}% | Len-F1={vit_len:.2f}% | Time={vit_t:.2f}s")

    # Plot Bayesian Optimization Loss Graph
    trials = np.arange(1, 16)
    best_so_far = np.minimum.accumulate(trial_losses)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(trials, trial_losses, "o--", color="#3498db", alpha=0.75, label="Trial CV Log Loss")
    ax1.plot(trials, best_so_far, "s-", color="#2ecc71", linewidth=2.2, label="Best-So-Far Log Loss")
    ax1.axhline(float(np.mean(untuned_cv_ll)), color="#e74c3c", linestyle=":", linewidth=1.8, label=f"Untuned RF ({np.mean(untuned_cv_ll):.4f})")
    ax1.axvline(5.5, color="gray", linestyle="--", alpha=0.5, label="Init -> GP-EI")
    ax1.set_title("Bayesian Optimization: CV Log Loss", fontweight="bold")
    ax1.set_xlabel("Optimization Trial", fontweight="bold")
    ax1.set_ylabel("5-Fold CV Log Loss", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=8.5)

    ax2.plot(trials, [f * 100 for f in trial_f1s], "o-", color="#9b59b6", linewidth=2, label="Trial CV Macro-F1 (%)")
    ax2.axhline(float(np.mean(untuned_cv_f1)) * 100, color="#e74c3c", linestyle=":", linewidth=1.8, label=f"Untuned RF ({np.mean(untuned_cv_f1)*100:.2f}%)")
    ax2.axvline(5.5, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Bayesian Optimization: CV Macro-F1", fontweight="bold")
    ax2.set_xlabel("Optimization Trial", fontweight="bold")
    ax2.set_ylabel("5-Fold CV Macro-F1 (%)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=8.5)

    plt.tight_layout()
    out_png = pkg_root / "report" / "rf_bayes_loss_curve.png"
    plt.savefig(out_png, dpi=180)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
