import math
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid

from baseline_hmm import (
    load_sample,
    project_to_meters,
    subsample_track,
    build_spatial_index,
    point_to_segments,
    build_adj,
    dijkstra_to_targets,
    reconstruct_node_path,
    hmm_map_match,
    evaluate_route,
)


def precompute_sample(sample_dir: Path, min_step_m: float, cand_radius: float = 60.0, max_cands: int = 12):
    nodes, arcs, track, route = load_sample(sample_dir)
    nodes_xy, track_xy = project_to_meters(nodes, track[:, :2])
    pts = subsample_track(track_xy, min_step_m=min_step_m)
    tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(nodes_xy, arcs)
    adj = build_adj(len(nodes_xy), arcs, arc_lens)

    candidates = []
    for p in pts:
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
        candidates.append((cands[order].tolist(), dists[order].tolist(), fracs[order].tolist()))

    # Precompute transition distances & intermediate arcs for each step t
    steps = []
    for t in range(1, len(pts)):
        d_euclid = math.hypot(pts[t, 0] - pts[t - 1, 0], pts[t, 1] - pts[t - 1, 1])
        max_route_dist = max(4.0 * d_euclid + 250.0, 500.0)
        prev_aids, _, prev_fracs = candidates[t - 1]
        curr_aids, _, curr_fracs = candidates[t]
        target_nodes = {int(arcs[a, 0]) for a in curr_aids}

        dijk_cache = {}
        for a_i in prev_aids:
            v_i = int(arcs[a_i, 1])
            if v_i not in dijk_cache:
                dijk_cache[v_i] = dijkstra_to_targets(adj, v_i, target_nodes, max_route_dist)

        edges = []  # (i, j, abs_diff, mid_arcs, connected)
        for i, (a_i, f_i) in enumerate(zip(prev_aids, prev_fracs)):
            v_i = int(arcs[a_i, 1])
            found, prev_map = dijk_cache[v_i]
            rem_i = (1.0 - f_i) * float(arc_lens[a_i])
            for j, (a_j, f_j) in enumerate(zip(curr_aids, curr_fracs)):
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
                edges.append((i, j, abs(r_dist - d_euclid), mid_arcs, conn))
        steps.append((len(curr_aids), edges))

    return candidates, steps, route, arc_lens


def fast_viterbi(precomputed, sigma_z: float, beta: float):
    candidates, steps, route, arc_lens = precomputed
    inv_2sig2 = -0.5 / (sigma_z * sigma_z)
    inv_beta = -1.0 / beta

    dp = [d * d * inv_2sig2 for d in candidates[0][1]]
    back = []

    for n_curr, edges in steps:
        t_idx = len(back) + 1
        curr_dists = candidates[t_idx][1]
        emits = [d * d * inv_2sig2 for d in curr_dists]

        new_dp = [-1e18] * n_curr
        step_back = [(0, []) for _ in range(n_curr)]
        for i, j, abs_diff, mid_arcs, _ in edges:
            if dp[i] <= -1e17:
                continue
            score = dp[i] + abs_diff * inv_beta + emits[j]
            if score > new_dp[j]:
                new_dp[j] = score
                step_back[j] = (i, mid_arcs)

        if max(new_dp) <= -1e17:
            best_i = max(range(len(dp)), key=lambda k: dp[k])
            for j in range(n_curr):
                new_dp[j] = dp[best_i] - 50.0 + emits[j]
                step_back[j] = (best_i, [])

        dp = new_dp
        back.append(step_back)

    T = len(candidates)
    cur_idx = max(range(len(dp)), key=lambda k: dp[k])
    chosen_arcs = [candidates[-1][0][cur_idx]]
    rev_segments = []
    for t in range(T - 2, -1, -1):
        prev_idx, mid_arcs = back[t][cur_idx]
        rev_segments.append(mid_arcs)
        chosen_arcs.append(candidates[t][0][prev_idx])
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

    return evaluate_route(full_route, route, arc_lens)


def main():
    root = Path(__file__).resolve().parent / "Student folder"
    train_dirs = sorted([d for d in (root / "train").iterdir() if d.is_dir()])
    test_dirs = sorted([d for d in (root / "student_test").iterdir() if d.is_dir()])

    param_grid = {
        "min_step_m": [15.0, 25.0],
        "sigma_z": [10.0, 15.0, 25.0],
        "beta": [10.0, 20.0, 35.0, 50.0],
    }
    grid = list(ParameterGrid(param_grid))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(train_dirs))

    print(f"5-Fold CV Grid Search on {len(train_dirs)} train samples ({len(grid)} configs)...\n")

    cache_by_step = {}
    for step_m in param_grid["min_step_m"]:
        t0 = time.perf_counter()
        print(f"Precomputing graph/Dijkstra tables for min_step_m={step_m}m...")
        cache_by_step[step_m] = [precompute_sample(d, min_step_m=step_m) for d in train_dirs]
        print(f"  Done in {time.perf_counter() - t0:.1f}s")

    print("\nRunning 5-fold CV over grid:")
    print(f"{'min_step':>8} {'sigma_z':>8} {'beta':>6} | {'CV Cnt-F1':>10} {'CV Len-F1':>10} {'Std Len-F1':>10}")
    print("-" * 64)

    results = []
    for params in grid:
        pre_list = cache_by_step[params["min_step_m"]]
        sample_metrics = [
            fast_viterbi(pre, sigma_z=params["sigma_z"], beta=params["beta"])
            for pre in pre_list
        ]

        fold_cnt_f1 = []
        fold_len_f1 = []
        for _, val_idx in splits:
            fold_cnt_f1.append(np.mean([sample_metrics[i]["f1_cnt"] for i in val_idx]))
            fold_len_f1.append(np.mean([sample_metrics[i]["f1_len"] for i in val_idx]))

        mean_cnt = float(np.mean(fold_cnt_f1))
        mean_len = float(np.mean(fold_len_f1))
        std_len = float(np.std(fold_len_f1))
        results.append((mean_len, mean_cnt, std_len, params))
        print(
            f"{params['min_step_m']:>8.1f} {params['sigma_z']:>8.1f} {params['beta']:>6.1f} | "
            f"{mean_cnt*100:>9.2f}% {mean_len*100:>9.2f}% {std_len*100:>9.2f}%"
        )

    results.sort(key=lambda x: x[0], reverse=True)
    best_len, best_cnt, best_std, best_params = results[0]
    print("-" * 64)
    print(f"Best CV params (by Len-F1): {best_params}")
    print(f"  5-Fold CV Len-F1: {best_len*100:.2f}% (+/- {best_std*100:.2f}%), Cnt-F1: {best_cnt*100:.2f}%\n")

    # Evaluate best params on held-out student_test
    print(f"Evaluating best CV params on {len(test_dirs)} student_test samples...")
    print(f"{'Sample':<10} {'Pred':>6} {'True':>6} {'Cnt-F1':>8} {'Len-P':>8} {'Len-R':>8} {'Len-F1':>8}")
    print("-" * 62)

    test_metrics = []
    for s in test_dirs:
        pre = precompute_sample(s, min_step_m=best_params["min_step_m"])
        m = fast_viterbi(pre, sigma_z=best_params["sigma_z"], beta=best_params["beta"])
        test_metrics.append(m)
        print(
            f"{s.name:<10} {m['pred_n']:>6} {m['true_n']:>6} "
            f"{m['f1_cnt']*100:>7.2f}% {m['prec_len']*100:>7.2f}% "
            f"{m['rec_len']*100:>7.2f}% {m['f1_len']*100:>7.2f}%"
        )

    print("-" * 62)
    print(
        f"{'MEAN':<10} {'':>6} {'':>6} "
        f"{np.mean([m['f1_cnt'] for m in test_metrics])*100:>7.2f}% "
        f"{np.mean([m['prec_len'] for m in test_metrics])*100:>7.2f}% "
        f"{np.mean([m['rec_len'] for m in test_metrics])*100:>7.2f}% "
        f"{np.mean([m['f1_len'] for m in test_metrics])*100:>7.2f}%"
    )


if __name__ == "__main__":
    main()
