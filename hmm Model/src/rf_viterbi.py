"""Pure RF + Viterbi Route Reconstruction (no analytical HMM) and RF + Naive Bayes."""

import math
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB

from baseline_hmm import (
    load_sample,
    project_to_meters,
    smooth_track,
    build_spatial_index,
    point_to_segments,
    build_adj,
    dijkstra_to_targets,
    reconstruct_node_path,
    evaluate_route,
)
from hybrid_hmm_rf import subsample_with_timestamps, compute_headings
from bayes_tune_rf import extract_rf_dataset


def rf_viterbi_reconstruct(
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    track_xy: np.ndarray,
    timestamps: np.ndarray,
    rf_clf,
    nb_clf=None,
    nb_weight: float = 0.0,
):
    """Viterbi route reconstruction scored solely by RF (and optional Naive Bayes), without HMM formulas."""
    smoothed_xy = smooth_track(track_xy, timestamps=timestamps, window=7, polyorder=2)
    pts, ts = subsample_with_timestamps(smoothed_xy, timestamps, min_step_m=20.0)
    tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(nodes_xy, arcs, sample_step_m=40.0)
    adj = build_adj(len(nodes_xy), arcs, arc_lens)

    n_nodes = len(nodes_xy)
    in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)
    out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)
    headings = compute_headings(pts)

    candidates = []
    for t, p in enumerate(pts):
        cands = np.array([], dtype=np.int32)
        for mult in (1.0, 2.5, 6.0):
            idxs = tree.query_ball_point(p, 60.0 * mult)
            if idxs:
                cands = np.unique(all_aids[idxs])
                break
        if len(cands) == 0:
            _, idxs = tree.query(p, k=min(12, len(all_aids)))
            cands = np.unique(all_aids[np.atleast_1d(idxs)])

        dists, fracs = point_to_segments(p, u_xy, diff, arc_lens, cands)
        order = np.argsort(dists)[:12]
        cands, dists, fracs = cands[order], dists[order], fracs[order]
        min_d = float(dists[0])
        hx, hy = headings[t]
        step_cands = []
        for rank, (aid, dist, frac) in enumerate(zip(cands.tolist(), dists.tolist(), fracs.tolist())):
            L = float(arc_lens[aid])
            if L > 1e-6 and (hx != 0.0 or hy != 0.0):
                cos_h = float(np.clip((diff[aid, 0] * hx + diff[aid, 1] * hy) / L, -1.0, 1.0))
                h_diff = math.acos(cos_h)
            else:
                cos_h, h_diff = 0.0, 0.0
            step_cands.append((aid, dist, frac, rank, dist - min_d, dist / max(min_d, 1.0), h_diff, cos_h, L))
        candidates.append(step_cands)

    T = len(pts)
    step_transitions = []
    all_feats = []

    for t in range(1, T):
        d_euclid = math.hypot(pts[t, 0] - pts[t - 1, 0], pts[t, 1] - pts[t - 1, 1])
        dt = max(float(ts[t] - ts[t - 1]), 1e-3)
        speed_mps = d_euclid / dt
        max_route_dist = max(4.0 * d_euclid + 250.0, 500.0)

        prev_cands = candidates[t - 1]
        curr_cands = candidates[t]
        target_nodes = {int(arcs[c[0], 0]) for c in curr_cands}

        dijk_cache = {}
        for c_i in prev_cands:
            v_i = int(arcs[c_i[0], 1])
            if v_i not in dijk_cache:
                dijk_cache[v_i] = dijkstra_to_targets(adj, v_i, target_nodes, max_route_dist)

        edges = []
        for i, c_i in enumerate(prev_cands):
            a_i, _, f_i = c_i[0], c_i[1], c_i[2]
            v_i = int(arcs[a_i, 1])
            found, prev_map = dijk_cache[v_i]
            for j, c_j in enumerate(curr_cands):
                a_j, dist_j, f_j, rank_j, d_diff_j, d_rat_j, h_diff_j, cos_h_j, L_j = c_j
                u_j, v_j = int(arcs[a_j, 0]), int(arcs[a_j, 1])

                if a_i == a_j and f_j >= f_i:
                    mid_arcs = []
                    gap_dist = 0.0
                    conn = True
                elif v_i == u_j:
                    mid_arcs = []
                    gap_dist = 0.0
                    conn = True
                elif u_j in found:
                    mid_arcs = reconstruct_node_path(prev_map, v_i, u_j)
                    gap_dist = float(math.hypot(nodes_xy[u_j, 0] - nodes_xy[v_i, 0], nodes_xy[u_j, 1] - nodes_xy[v_i, 1]))
                    conn = True
                else:
                    mid_arcs = []
                    gap_dist = float(math.hypot(nodes_xy[u_j, 0] - nodes_xy[v_i, 0], nodes_xy[u_j, 1] - nodes_xy[v_i, 1]))
                    conn = False

                feat = [
                    dist_j, float(rank_j), d_diff_j, d_rat_j, h_diff_j, cos_h_j,
                    speed_mps, d_euclid, dt, L_j, f_j,
                    float(in_deg[u_j]), float(out_deg[v_j]),
                    1.0 if a_i == a_j else 0.0, 1.0 if v_i == u_j else 0.0, gap_dist,
                ]
                feat_idx = len(all_feats)
                all_feats.append(feat)
                edges.append((i, j, mid_arcs, feat_idx, conn))
        step_transitions.append(edges)

    X_arr = np.asarray(all_feats, dtype=np.float32)
    rf_logps = np.log(np.clip(rf_clf.predict_proba(X_arr)[:, 1], 1e-6, 1.0 - 1e-6))
    if nb_clf is not None and nb_weight > 0.0:
        nb_logps = np.log(np.clip(nb_clf.predict_proba(X_arr)[:, 1], 1e-6, 1.0 - 1e-6))
        scores = (rf_logps + nb_weight * nb_logps).tolist()
    else:
        scores = rf_logps.tolist()

    # Viterbi over RF (+ optional NB) log-probabilities only (no analytical HMM score)
    dp = [0.0] * len(candidates[0])
    back = []
    for t_idx, edges in enumerate(step_transitions, start=1):
        n_curr = len(candidates[t_idx])
        new_dp = [-1e18] * n_curr
        step_back = [(0, []) for _ in range(n_curr)]
        for i, j, mid_arcs, feat_idx, conn in edges:
            # Small disconnection penalty so Viterbi prefers connected graph paths
            sc = dp[i] + scores[feat_idx] + (0.0 if conn else -10.0)
            if sc > new_dp[j]:
                new_dp[j] = sc
                step_back[j] = (i, mid_arcs)
        dp = new_dp
        back.append(step_back)

    cur_idx = max(range(len(dp)), key=lambda k: dp[k])
    chosen_arcs = [candidates[-1][cur_idx][0]]
    rev_segments = []
    for t in range(T - 2, -1, -1):
        prev_idx, mid_arcs = back[t][cur_idx]
        rev_segments.append(mid_arcs)
        chosen_arcs.append(candidates[t][prev_idx][0])
        cur_idx = prev_idx

    chosen_arcs.reverse()
    rev_segments.reverse()

    full_route = [chosen_arcs[0]]
    for t in range(len(rev_segments)):
        for a in rev_segments[t]:
            if a != full_route[-1]:
                full_route.append(a)
        if chosen_arcs[t + 1] != full_route[-1]:
            full_route.append(chosen_arcs[t + 1])
    return full_route, arc_lens


def main():
    pkg_root = Path(__file__).resolve().parent.parent
    data_root = pkg_root.parent
    rf_bundle = joblib.load(pkg_root / "models" / "rf_bayes_tuned.joblib")
    rf_clf = rf_bundle["model"]

    # Fit GaussianNB on the same 25 train samples for RF + Naive Bayes comparison
    all_train = sorted([d for d in (data_root / "Student folder" / "train").iterdir() if d.is_dir()])
    train_dirs = [d for d in all_train if (d / f"{d.name}.arcs").stat().st_size < 2_500_000][::2][:25]
    X_tr, y_tr, _ = extract_rf_dataset(train_dirs, max_pts_per_sample=70)
    nb_clf = GaussianNB()
    nb_clf.fit(X_tr, y_tr)

    student_samples = sorted([d for d in (data_root / "Student folder" / "student_test").iterdir() if d.is_dir()])
    eval_dir = data_root / "evaluator_test" / "evaluator_test"
    if not eval_dir.exists():
        eval_dir = data_root / "evaluator_test"
    eval_samples = sorted([d for d in eval_dir.iterdir() if d.is_dir()])

    for label, nb, nw in [("Pure RF (Bayes-Opt) + Viterbi", None, 0.0), ("RF + NaiveBayes + Viterbi", nb_clf, 0.5)]:
        for split_name, samples in [("student_test", student_samples), ("evaluator_test", eval_samples)]:
            cnts, lens = [], []
            tp = fp = fn = tn = exact = 0
            for s in samples:
                n, a, t, r = load_sample(s)
                xy_n, xy_t = project_to_meters(n, t[:, :2])
                pred, alens = rf_viterbi_reconstruct(xy_n, a, xy_t, t[:, 2], rf_clf, nb_clf=nb, nb_weight=nw)
                m = evaluate_route(pred, r, alens)
                cnts.append(m["f1_cnt"] * 100)
                lens.append(m["f1_len"] * 100)
                ps, ts = set(pred), set(r)
                s_tp = len(ps & ts); s_fp = len(ps - ts); s_fn = len(ts - ps)
                tp += s_tp; fp += s_fp; fn += s_fn; tn += len(a) - (s_tp + s_fp + s_fn)
                if pred == r:
                    exact += 1
            top1 = (tp + tn) / (tp + tn + fp + fn) * 100.0
            print(f"{label:<28} | {split_name:<14} | Cnt-F1={np.mean(cnts):.2f}% | Len-F1={np.mean(lens):.2f}% | Top-1={top1:.4f}% | Exact={exact}/{len(samples)}")


if __name__ == "__main__":
    main()
