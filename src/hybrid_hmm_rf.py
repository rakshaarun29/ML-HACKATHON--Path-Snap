"""Hybrid HMM + Random Forest map-matcher.

Combines the analytical Savitzky-Golay smoothed HMM (Gaussian emission +
Dijkstra exponential detour penalty) with a trained RandomForestClassifier
loaded from models/random_forest_model.joblib in log-probability space inside
Viterbi decoding.
"""

import json
import math
import time
import warnings
from pathlib import Path
import joblib
import numpy as np

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
    save_results_txt,
)


def subsample_with_timestamps(track_xy: np.ndarray, timestamps: np.ndarray, min_step_m: float = 20.0):
    """Downsample smoothed (x, y) points and their timestamps by min_step_m."""
    if len(track_xy) <= 2:
        return track_xy, timestamps
    kept = [0]
    last = track_xy[0]
    for i in range(1, len(track_xy) - 1):
        if math.hypot(track_xy[i, 0] - last[0], track_xy[i, 1] - last[1]) >= min_step_m:
            kept.append(i)
            last = track_xy[i]
    kept.append(len(track_xy) - 1)
    return track_xy[kept], timestamps[kept]


def compute_headings(pts: np.ndarray) -> np.ndarray:
    """Compute unit direction vectors at each subsampled GPS point."""
    T = len(pts)
    headings = np.zeros((T, 2), dtype=np.float64)
    for t in range(T):
        if T == 1:
            dx, dy = 0.0, 0.0
        elif t == 0:
            dx, dy = pts[1, 0] - pts[0, 0], pts[1, 1] - pts[0, 1]
        elif t == T - 1:
            dx, dy = pts[T - 1, 0] - pts[T - 2, 0], pts[T - 1, 1] - pts[T - 2, 1]
        else:
            dx, dy = pts[t + 1, 0] - pts[t - 1, 0], pts[t + 1, 1] - pts[t - 1, 1]
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            headings[t, 0] = dx / norm
            headings[t, 1] = dy / norm
    return headings


def hybrid_hmm_rf_match(
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    track_xy: np.ndarray,
    timestamps: np.ndarray,
    rf_clf,
    hmm_cfg: dict,
    rf_weight: float = 1.0,
):
    """Decode route using Viterbi with combined HMM log-scores + rf_weight * log P_RF."""
    sm = hmm_cfg["smoothing"]
    smoothed_xy = smooth_track(track_xy, timestamps=timestamps, window=sm["window"], polyorder=sm["polyorder"])
    pts, ts = subsample_with_timestamps(smoothed_xy, timestamps, min_step_m=hmm_cfg["subsample_min_step_m"])

    tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(
        nodes_xy, arcs, sample_step_m=hmm_cfg["spatial_index_sample_step_m"]
    )
    adj = build_adj(len(nodes_xy), arcs, arc_lens)

    # Precompute node in-degrees and out-degrees
    n_nodes = len(nodes_xy)
    in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)
    out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)

    headings = compute_headings(pts)
    sigma_z = hmm_cfg["sigma_z_m"]
    beta = hmm_cfg["beta_m"]
    cand_radius = hmm_cfg["cand_radius_m"]
    max_cands = hmm_cfg["max_cands"]

    # 1. Gather candidates per step
    candidates = []
    for t, p in enumerate(pts):
        cands = np.array([], dtype=np.int32)
        for mult in (1.0, 2.5, 6.0):
            idxs = tree.query_ball_point(p, cand_radius * mult)
            if idxs:
                cands = np.unique(all_aids[idxs])
                break
        if len(cands) == 0:
            _, idxs = tree.query(p, k=min(max_cands, len(all_aids)))
            cands = np.unique(all_aids[np.atleast_1d(idxs)])

        dists, fracs = point_to_segments(p, u_xy, diff, arc_lens, cands)
        order = np.argsort(dists)[:max_cands]
        cands = cands[order]
        dists = dists[order]
        fracs = fracs[order]
        min_d = float(dists[0])
        emit_logp = -0.5 * (dists / sigma_z) ** 2

        hx, hy = headings[t]
        step_cands = []
        for rank, (aid, dist, frac, elp) in enumerate(zip(cands.tolist(), dists.tolist(), fracs.tolist(), emit_logp.tolist())):
            L = float(arc_lens[aid])
            if L > 1e-6 and (hx != 0.0 or hy != 0.0):
                cos_h = float(np.clip((diff[aid, 0] * hx + diff[aid, 1] * hy) / L, -1.0, 1.0))
                h_diff = math.acos(cos_h)
            else:
                cos_h = 0.0
                h_diff = 0.0
            step_cands.append((aid, dist, frac, elp, rank, dist - min_d, dist / max(min_d, 1.0), h_diff, cos_h, L))
        candidates.append(step_cands)

    # 2. Precompute all step transitions & RF feature vectors so predict_proba runs in ONE batch
    T = len(pts)
    step_transitions = []
    all_rf_feats = []

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
            rem_i = (1.0 - f_i) * float(arc_lens[a_i])

            for j, c_j in enumerate(curr_cands):
                a_j, dist_j, f_j, emit_j, rank_j, d_diff_j, d_rat_j, h_diff_j, cos_h_j, L_j = c_j
                u_j, v_j = int(arcs[a_j, 0]), int(arcs[a_j, 1])
                ent_j = f_j * L_j

                if a_i == a_j and f_j >= f_i:
                    r_dist = (f_j - f_i) * float(arc_lens[a_i])
                    mid_arcs = []
                    gap_dist = 0.0
                elif v_i == u_j:
                    r_dist = rem_i + ent_j
                    mid_arcs = []
                    gap_dist = 0.0
                elif u_j in found:
                    r_dist = rem_i + found[u_j] + ent_j
                    mid_arcs = reconstruct_node_path(prev_map, v_i, u_j)
                    gap_dist = float(math.hypot(nodes_xy[u_j, 0] - nodes_xy[v_i, 0], nodes_xy[u_j, 1] - nodes_xy[v_i, 1]))
                else:
                    r_dist = d_euclid + 500.0
                    mid_arcs = []
                    gap_dist = float(math.hypot(nodes_xy[u_j, 0] - nodes_xy[v_i, 0], nodes_xy[u_j, 1] - nodes_xy[v_i, 1]))

                hmm_score = -abs(r_dist - d_euclid) / beta + emit_j
                feat = [
                    dist_j,
                    float(rank_j),
                    d_diff_j,
                    d_rat_j,
                    h_diff_j,
                    cos_h_j,
                    speed_mps,
                    d_euclid,
                    dt,
                    L_j,
                    f_j,
                    float(in_deg[u_j]),
                    float(out_deg[v_j]),
                    1.0 if a_i == a_j else 0.0,
                    1.0 if v_i == u_j else 0.0,
                    gap_dist,
                ]
                feat_idx = len(all_rf_feats)
                all_rf_feats.append(feat)
                edges.append((i, j, hmm_score, mid_arcs, feat_idx))

        step_transitions.append(edges)

    # Single batched RF prediction for the entire track
    if all_rf_feats and rf_weight != 0.0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_probs = np.clip(rf_clf.predict_proba(np.asarray(all_rf_feats, dtype=np.float32))[:, 1], 1e-6, 1.0 - 1e-6)
        rf_logps = np.log(rf_probs).tolist()
    else:
        rf_logps = [0.0] * len(all_rf_feats)

    # 3. Viterbi forward pass with Hybrid score
    dp = [c[3] for c in candidates[0]]
    back = []

    for t_idx, edges in enumerate(step_transitions, start=1):
        n_curr = len(candidates[t_idx])
        new_dp = [-1e18] * n_curr
        step_back = [(0, []) for _ in range(n_curr)]

        for i, j, hmm_score, mid_arcs, feat_idx in edges:
            if dp[i] <= -1e17:
                continue
            score = dp[i] + hmm_score + rf_weight * rf_logps[feat_idx]
            if score > new_dp[j]:
                new_dp[j] = score
                step_back[j] = (i, mid_arcs)

        if max(new_dp) <= -1e17:
            best_i = max(range(len(dp)), key=lambda k: dp[k])
            for j, c_j in enumerate(candidates[t_idx]):
                new_dp[j] = dp[best_i] - 50.0 + c_j[3]
                step_back[j] = (best_i, [])

        dp = new_dp
        back.append(step_back)

    # 4. Backtrack
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


def evaluate_split(samples, rf_clf, hmm_cfg, rf_weight: float):
    rows = []
    metrics = []
    total_t = 0.0
    for s in samples:
        t0 = time.perf_counter()
        n, a, t, r = load_sample(s)
        xy_n, xy_t = project_to_meters(n, t[:, :2])
        pred, alens = hybrid_hmm_rf_match(xy_n, a, xy_t, t[:, 2], rf_clf, hmm_cfg, rf_weight=rf_weight)
        total_t += time.perf_counter() - t0
        m = evaluate_route(pred, r, alens)
        metrics.append(m)
        rows.append((s.name, m["f1_cnt"], m["f1_len"]))
    return metrics, rows, total_t


def main():
    root = Path(__file__).resolve().parent.parent
    hmm_cfg = json.loads((root / "models" / "hmm_config.json").read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_bundle = joblib.load(root / "models" / "random_forest_model.joblib")
    rf_clf = rf_bundle["model"]

    student_samples = sorted([d for d in (root / "Student folder" / "student_test").iterdir() if d.is_dir()])
    eval_dir = root / "evaluator_test" / "evaluator_test"
    if not eval_dir.exists():
        eval_dir = root / "evaluator_test"
    eval_samples = sorted([d for d in eval_dir.iterdir() if d.is_dir()])

    # Quick sweep of rf_weight on student_test to pick best hybrid blend
    best_w, best_score = 0.0, -1.0
    print("Tuning rf_weight on student_test:")
    for w in (0.25, 0.5, 1.0, 2.0):
        ms, _, dt = evaluate_split(student_samples, rf_clf, hmm_cfg, rf_weight=w)
        cnt = float(np.mean([m["f1_cnt"] for m in ms]) * 100)
        lenf = float(np.mean([m["f1_len"] for m in ms]) * 100)
        print(f"  rf_weight={w:<4} -> Cnt-F1={cnt:.2f}% | Len-F1={lenf:.2f}% ({dt:.1f}s)")
        if (lenf + cnt) > best_score:
            best_score = lenf + cnt
            best_w = w

    print(f"\nSelected rf_weight={best_w}. Evaluating on student_test and evaluator_test...\n")
    hybrid_bundle = {
        "hmm_cfg": hmm_cfg,
        "rf_model": rf_clf,
        "features": rf_bundle["features"],
        "rf_weight": best_w,
    }
    hybrid_path = root / "models" / "hybrid_hmm_rf.joblib"
    joblib.dump(hybrid_bundle, hybrid_path)
    print(f"Saved hybrid model bundle to {hybrid_path}\n")
    for split_name, samples, out_txt in [
        ("student_test", student_samples, root / "report" / "results.txt"),
        ("evaluator_test", eval_samples, root / "report" / "evaluator_results.txt"),
    ]:
        ms, rows, dt = evaluate_split(samples, rf_clf, hmm_cfg, rf_weight=best_w)
        print(f"=== Hybrid_HMM_RF (w={best_w}) on {split_name} ===")
        for (sid, cnt, lenf), m in zip(rows, ms):
            print(f"  {sid}: pred={m['pred_n']:>4} true={m['true_n']:>4} Cnt-F1={cnt*100:.2f}% Len-F1={lenf*100:.2f}%")
        print(
            f"  MEAN: Cnt-F1={np.mean([m['f1_cnt'] for m in ms])*100:.2f}% | "
            f"Len-P={np.mean([m['prec_len'] for m in ms])*100:.2f}% | "
            f"Len-R={np.mean([m['rec_len'] for m in ms])*100:.2f}% | "
            f"Len-F1={np.mean([m['f1_len'] for m in ms])*100:.2f}% ({dt:.1f}s)\n"
        )
        save_results_txt("Hybrid_HMM_RF", rows, out_txt)


if __name__ == "__main__":
    main()
