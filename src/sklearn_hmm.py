"""ML-augmented HMM map-matcher using scikit-learn HistGradientBoostingClassifiers.

Trains two binary classifiers on the 80 train samples:
  1. emit_clf: predicts whether a candidate arc at step t belongs to the true route.
  2. trans_clf: predicts whether a candidate transition (a_i -> a_j) is valid.
During inference, predicted log-probabilities replace the analytical Gaussian
and exponential scores inside Viterbi decoding.
"""

import math
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from baseline_hmm import (
    load_sample,
    project_to_meters,
    subsample_track,
    build_spatial_index,
    point_to_segments,
    build_adj,
    dijkstra_to_targets,
    reconstruct_node_path,
    evaluate_route,
    save_results_txt,
)


def compute_track_headings(pts: np.ndarray) -> np.ndarray:
    """Compute unit direction vectors at each subsampled GPS point using finite differences."""
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


def extract_candidates(pts, tree, all_aids, u_xy, diff, arc_lens, cand_radius=60.0, max_cands=10):
    """Find top-k candidate arcs and build 6 emission features per candidate at each step."""
    headings = compute_track_headings(pts)
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

        hx, hy = headings[t]
        step_cands = []
        for rank, (aid, dist, frac) in enumerate(zip(cands.tolist(), dists.tolist(), fracs.tolist())):
            L = float(arc_lens[aid])
            # Cosine alignment between GPS heading and directed arc vector
            if L > 1e-6 and (hx != 0.0 or hy != 0.0):
                cos_h = (diff[aid, 0] * hx + diff[aid, 1] * hy) / L
            else:
                cos_h = 0.0
            is_end = 1.0 if (frac <= 1e-6 or frac >= 1.0 - 1e-6) else 0.0
            feat = [dist, float(rank), cos_h, L, frac, is_end]
            step_cands.append((aid, frac, feat))
        candidates.append(step_cands)
    return candidates


def transition_features(arcs, diff, arc_lens, a_i, f_i, a_j, f_j, d_euclid, r_dist, connected):
    """Build 9 transition features for moving from candidate arc a_i to candidate arc a_j."""
    u_i, v_i = int(arcs[a_i, 0]), int(arcs[a_i, 1])
    u_j, v_j = int(arcs[a_j, 0]), int(arcs[a_j, 1])
    L_i = float(arc_lens[a_i])
    L_j = float(arc_lens[a_j])
    if L_i > 1e-6 and L_j > 1e-6:
        cos_turn = float((diff[a_i, 0] * diff[a_j, 0] + diff[a_i, 1] * diff[a_j, 1]) / (L_i * L_j))
    else:
        cos_turn = 0.0
    abs_diff = abs(r_dist - d_euclid)
    rel_diff = abs_diff / max(d_euclid, 10.0)
    same_arc = 1.0 if a_i == a_j else 0.0
    direct_conn = 1.0 if v_i == u_j else 0.0
    is_uturn = 1.0 if (u_i == v_j and v_i == u_j) else 0.0
    return [
        abs_diff,
        rel_diff,
        r_dist,
        d_euclid,
        same_arc,
        direct_conn,
        is_uturn,
        1.0 if connected else 0.0,
        cos_turn,
    ]


def collect_training_data(train_dir: Path, max_samples: int = 80):
    """Extract emission and transition feature matrices and binary labels from train samples."""
    samples = sorted([d for d in train_dir.iterdir() if d.is_dir()])[:max_samples]
    X_emit, y_emit = [], []
    X_trans, y_trans = [], []

    for s in samples:
        nodes, arcs, track, route = load_sample(s)
        true_set = set(route)
        nodes_xy, track_xy = project_to_meters(nodes, track[:, :2])
        pts = subsample_track(track_xy, min_step_m=20.0)
        tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(nodes_xy, arcs)
        adj = build_adj(len(nodes_xy), arcs, arc_lens)
        candidates = extract_candidates(pts, tree, all_aids, u_xy, diff, arc_lens)

        # Label each candidate arc 1 if it appears in the ground-truth route
        for step_cands in candidates:
            for aid, _, feat in step_cands:
                X_emit.append(feat)
                y_emit.append(1 if aid in true_set else 0)

        # Label each candidate pair (a_i -> a_j) 1 if both are in true_set and connected
        for t in range(1, len(pts)):
            d_euclid = math.hypot(pts[t, 0] - pts[t - 1, 0], pts[t, 1] - pts[t - 1, 1])
            max_route_dist = max(4.0 * d_euclid + 250.0, 500.0)
            prev_cands = candidates[t - 1]
            curr_cands = candidates[t]
            target_nodes = {int(arcs[a, 0]) for a, _, _ in curr_cands}

            dijk_cache = {}
            for a_i, _, _ in prev_cands:
                v_i = int(arcs[a_i, 1])
                if v_i not in dijk_cache:
                    dijk_cache[v_i] = dijkstra_to_targets(adj, v_i, target_nodes, max_route_dist)

            for a_i, f_i, _ in prev_cands:
                v_i = int(arcs[a_i, 1])
                found, _ = dijk_cache[v_i]
                rem_i = (1.0 - f_i) * float(arc_lens[a_i])
                in_i = a_i in true_set

                for a_j, f_j, _ in curr_cands:
                    u_j = int(arcs[a_j, 0])
                    ent_j = f_j * float(arc_lens[a_j])
                    if a_i == a_j and f_j >= f_i:
                        r_dist = (f_j - f_i) * float(arc_lens[a_i])
                        conn = True
                    elif v_i == u_j:
                        r_dist = rem_i + ent_j
                        conn = True
                    elif u_j in found:
                        r_dist = rem_i + found[u_j] + ent_j
                        conn = True
                    else:
                        r_dist = d_euclid + 500.0
                        conn = False

                    feat = transition_features(arcs, diff, arc_lens, a_i, f_i, a_j, f_j, d_euclid, r_dist, conn)
                    X_trans.append(feat)
                    y_trans.append(1 if (in_i and (a_j in true_set) and conn) else 0)

    return (
        np.asarray(X_emit, dtype=np.float32),
        np.asarray(y_emit, dtype=np.int32),
        np.asarray(X_trans, dtype=np.float32),
        np.asarray(y_trans, dtype=np.int32),
    )


def sklearn_hmm_match(nodes_xy, arcs, track_xy, emit_clf, trans_clf):
    """Run Viterbi decoding using log-probabilities from trained sklearn classifiers."""
    pts = subsample_track(track_xy, min_step_m=20.0)
    tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(nodes_xy, arcs)
    adj = build_adj(len(nodes_xy), arcs, arc_lens)
    candidates = extract_candidates(pts, tree, all_aids, u_xy, diff, arc_lens)

    # Batch-predict emission log-probabilities for all steps at once
    all_emit_feats = np.asarray([c[2] for step in candidates for c in step], dtype=np.float32)
    emit_probs = np.clip(emit_clf.predict_proba(all_emit_feats)[:, 1], 1e-6, 1.0 - 1e-6)
    emit_logps = np.log(emit_probs)

    emit_by_step = []
    idx = 0
    for step in candidates:
        k = len(step)
        emit_by_step.append(emit_logps[idx : idx + k].tolist())
        idx += k

    T = len(pts)
    dp = list(emit_by_step[0])
    back = []

    for t in range(1, T):
        d_euclid = math.hypot(pts[t, 0] - pts[t - 1, 0], pts[t, 1] - pts[t - 1, 1])
        max_route_dist = max(4.0 * d_euclid + 250.0, 500.0)
        prev_cands = candidates[t - 1]
        curr_cands = candidates[t]
        target_nodes = {int(arcs[a, 0]) for a, _, _ in curr_cands}

        dijk_cache = {}
        for a_i, _, _ in prev_cands:
            v_i = int(arcs[a_i, 1])
            if v_i not in dijk_cache:
                dijk_cache[v_i] = dijkstra_to_targets(adj, v_i, target_nodes, max_route_dist)

        trans_feats = []
        pair_meta = []
        for i, (a_i, f_i, _) in enumerate(prev_cands):
            v_i = int(arcs[a_i, 1])
            found, prev_map = dijk_cache[v_i]
            rem_i = (1.0 - f_i) * float(arc_lens[a_i])
            for j, (a_j, f_j, _) in enumerate(curr_cands):
                u_j = int(arcs[a_j, 0])
                ent_j = f_j * float(arc_lens[a_j])
                if a_i == a_j and f_j >= f_i:
                    r_dist = (f_j - f_i) * float(arc_lens[a_i])
                    mid_arcs = []
                    conn = True
                elif v_i == u_j:
                    r_dist = rem_i + ent_j
                    mid_arcs = []
                    conn = True
                elif u_j in found:
                    r_dist = rem_i + found[u_j] + ent_j
                    mid_arcs = reconstruct_node_path(prev_map, v_i, u_j)
                    conn = True
                else:
                    r_dist = d_euclid + 500.0
                    mid_arcs = []
                    conn = False
                trans_feats.append(
                    transition_features(arcs, diff, arc_lens, a_i, f_i, a_j, f_j, d_euclid, r_dist, conn)
                )
                pair_meta.append((i, j, mid_arcs))

        t_probs = np.clip(trans_clf.predict_proba(np.asarray(trans_feats, dtype=np.float32))[:, 1], 1e-6, 1.0 - 1e-6)
        t_logps = np.log(t_probs)

        new_dp = [-1e18] * len(curr_cands)
        step_back = [(0, []) for _ in range(len(curr_cands))]
        for (i, j, mid_arcs), t_lp in zip(pair_meta, t_logps.tolist()):
            score = dp[i] + t_lp + emit_by_step[t][j]
            if score > new_dp[j]:
                new_dp[j] = score
                step_back[j] = (i, mid_arcs)

        dp = new_dp
        back.append(step_back)

    # Backtrack Viterbi pointers and reconstruct full arc sequence
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
    """Train HGBC emission/transition classifiers on train/ and evaluate on student_test/."""
    proj_root = Path(__file__).resolve().parent.parent
    root = proj_root / "Student folder"
    t0 = time.perf_counter()
    print("Extracting features from train (80 samples)...")
    X_emit, y_emit, X_trans, y_trans = collect_training_data(root / "train", max_samples=80)
    print(f"  Done in {time.perf_counter() - t0:.1f}s | emit={X_emit.shape}, trans={X_trans.shape}")

    t1 = time.perf_counter()
    emit_clf = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)
    emit_clf.fit(X_emit, y_emit)
    trans_clf = HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42)
    trans_clf.fit(X_trans, y_trans)
    print(f"Trained HistGradientBoostingClassifiers in {time.perf_counter() - t1:.1f}s\n")

    models_dir = proj_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "sklearn_hgbc_hmm.joblib"
    joblib.dump({"emit_clf": emit_clf, "trans_clf": trans_clf}, model_path)
    print(f"Saved trained sklearn model to {model_path}\n")

    test_samples = sorted([d for d in (root / "student_test").iterdir() if d.is_dir()])
    print(f"Evaluating sklearn HMM on {len(test_samples)} student_test samples...\n")
    print(f"{'Sample':<10} {'Pred':>6} {'True':>6} {'Cnt-F1':>8} {'Len-P':>8} {'Len-R':>8} {'Len-F1':>8} {'Time':>7}")
    print("-" * 70)

    metrics = []
    result_rows = []
    for s in test_samples:
        ts = time.perf_counter()
        nodes, arcs, track, route = load_sample(s)
        nodes_xy, track_xy = project_to_meters(nodes, track[:, :2])
        pred_route, arc_lens = sklearn_hmm_match(nodes_xy, arcs, track_xy, emit_clf, trans_clf)
        dt = time.perf_counter() - ts
        m = evaluate_route(pred_route, route, arc_lens)
        metrics.append(m)
        result_rows.append((s.name, m["f1_cnt"], m["f1_len"]))
        print(
            f"{s.name:<10} {m['pred_n']:>6} {m['true_n']:>6} "
            f"{m['f1_cnt']*100:>7.2f}% {m['prec_len']*100:>7.2f}% "
            f"{m['rec_len']*100:>7.2f}% {m['f1_len']*100:>7.2f}% {dt:>6.2f}s"
        )

    print("-" * 70)
    mean_cnt_f1 = sum(m["f1_cnt"] for m in metrics) / len(metrics)
    mean_len_p = sum(m["prec_len"] for m in metrics) / len(metrics)
    mean_len_r = sum(m["rec_len"] for m in metrics) / len(metrics)
    mean_len_f1 = sum(m["f1_len"] for m in metrics) / len(metrics)
    print(
        f"{'MEAN':<10} {'':>6} {'':>6} "
        f"{mean_cnt_f1*100:>7.2f}% {mean_len_p*100:>7.2f}% "
        f"{mean_len_r*100:>7.2f}% {mean_len_f1*100:>7.2f}%"
    )

    out_txt = proj_root / "report" / "results.txt"
    save_results_txt("sklearn_HGBC", result_rows, out_txt)
    print(f"\nSaved accuracies to {out_txt}")


if __name__ == "__main__":
    main()
