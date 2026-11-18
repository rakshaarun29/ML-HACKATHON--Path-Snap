import math
import time
import heapq
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


def load_sample(sample_dir: Path):
    sid = sample_dir.name
    nodes = np.loadtxt(sample_dir / f"{sid}.nodes", dtype=np.float64)
    arcs = np.loadtxt(sample_dir / f"{sid}.arcs", dtype=np.int32)
    track = np.loadtxt(sample_dir / f"{sid}.track", dtype=np.float64)
    route = np.loadtxt(sample_dir / f"{sid}.route", dtype=np.int32).tolist()
    if isinstance(route, int):
        route = [route]
    return nodes, arcs, track, route


def project_to_meters(nodes_lonlat: np.ndarray, track_lonlat: np.ndarray):
    lon0 = float(np.mean(nodes_lonlat[:, 0]))
    lat0 = float(np.mean(nodes_lonlat[:, 1]))
    scale_x = math.cos(math.radians(lat0)) * 111320.0
    scale_y = 111320.0

    nodes_xy = np.empty_like(nodes_lonlat)
    nodes_xy[:, 0] = (nodes_lonlat[:, 0] - lon0) * scale_x
    nodes_xy[:, 1] = (nodes_lonlat[:, 1] - lat0) * scale_y

    track_xy = np.empty((len(track_lonlat), 2), dtype=np.float64)
    track_xy[:, 0] = (track_lonlat[:, 0] - lon0) * scale_x
    track_xy[:, 1] = (track_lonlat[:, 1] - lat0) * scale_y
    return nodes_xy, track_xy


def subsample_track(track_xy: np.ndarray, min_step_m: float = 20.0):
    if len(track_xy) <= 2:
        return track_xy
    kept = [0]
    last = track_xy[0]
    for i in range(1, len(track_xy) - 1):
        d = math.hypot(track_xy[i, 0] - last[0], track_xy[i, 1] - last[1])
        if d >= min_step_m:
            kept.append(i)
            last = track_xy[i]
    kept.append(len(track_xy) - 1)
    return track_xy[kept]


def build_spatial_index(nodes_xy: np.ndarray, arcs: np.ndarray, sample_step_m: float = 40.0):
    u_xy = nodes_xy[arcs[:, 0]]
    v_xy = nodes_xy[arcs[:, 1]]
    diff = v_xy - u_xy
    arc_lens = np.hypot(diff[:, 0], diff[:, 1])

    pts_list = []
    arc_ids_list = []
    # Sample midpoints + intermediate points for long arcs
    counts = np.maximum(1, np.ceil(arc_lens / sample_step_m).astype(np.int32))
    max_k = int(counts.max())
    # Vectorized for k=1, loop only for longer arcs
    mids = 0.5 * (u_xy + v_xy)
    pts_list.append(mids)
    arc_ids_list.append(np.arange(len(arcs), dtype=np.int32))

    long_idx = np.where(counts > 1)[0]
    for aid in long_idx:
        k = counts[aid]
        ts = np.linspace(0.0, 1.0, k + 1)[1:-1]  # endpoints already covered by neighbors/mid
        pts = u_xy[aid] + ts[:, None] * diff[aid]
        pts_list.append(pts)
        arc_ids_list.append(np.full(len(ts), aid, dtype=np.int32))

    # Also include start/end nodes slightly inset
    pts_list.append(u_xy)
    arc_ids_list.append(np.arange(len(arcs), dtype=np.int32))
    pts_list.append(v_xy)
    arc_ids_list.append(np.arange(len(arcs), dtype=np.int32))

    all_pts = np.vstack(pts_list)
    all_aids = np.concatenate(arc_ids_list)
    tree = cKDTree(all_pts)
    return tree, all_aids, u_xy, v_xy, diff, arc_lens


def point_to_segments(p: np.ndarray, u_xy: np.ndarray, diff: np.ndarray, arc_lens: np.ndarray, cand_arcs: np.ndarray):
    u = u_xy[cand_arcs]
    d = diff[cand_arcs]
    L = arc_lens[cand_arcs]
    L2 = L * L
    ap = p - u
    dot = ap[:, 0] * d[:, 0] + ap[:, 1] * d[:, 1]
    frac = np.where(L2 > 1e-9, np.clip(dot / np.maximum(L2, 1e-9), 0.0, 1.0), 0.0)
    proj = u + frac[:, None] * d
    dist = np.hypot(p[0] - proj[:, 0], p[1] - proj[:, 1])
    return dist, frac


def build_adj(num_nodes: int, arcs: np.ndarray, arc_lens: np.ndarray):
    adj = [[] for _ in range(num_nodes)]
    for aid, (u, v) in enumerate(arcs):
        adj[u].append((v, float(arc_lens[aid]), aid))
    return adj


def dijkstra_to_targets(adj, start_node: int, targets: set, max_dist: float):
    if not targets:
        return {}, {}
    dist = {start_node: 0.0}
    prev = {}
    pq = [(0.0, start_node)]
    found = {}
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u in targets and u not in found:
            found[u] = d
            if len(found) == len(targets):
                break
        if d > max_dist:
            break
        for v, w, aid in adj[u]:
            nd = d + w
            if nd <= max_dist and nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = (u, aid)
                heapq.heappush(pq, (nd, v))
    return found, prev


def reconstruct_node_path(prev: dict, start_node: int, end_node: int):
    path = []
    cur = end_node
    while cur != start_node:
        p, aid = prev[cur]
        path.append(aid)
        cur = p
    path.reverse()
    return path


def hmm_map_match(
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    track_xy: np.ndarray,
    sigma_z: float = 15.0,
    beta: float = 25.0,
    cand_radius: float = 60.0,
    max_cands: int = 12,
):
    pts = subsample_track(track_xy, min_step_m=20.0)
    tree, all_aids, u_xy, v_xy, diff, arc_lens = build_spatial_index(nodes_xy, arcs)
    adj = build_adj(len(nodes_xy), arcs, arc_lens)

    # Gather candidates for each point
    candidates = []
    for p in pts:
        r = cand_radius
        cands = np.array([], dtype=np.int32)
        for mult in (1.0, 2.5, 6.0):
            idxs = tree.query_ball_point(p, r * mult)
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
        emit_logp = -0.5 * (dists / sigma_z) ** 2
        candidates.append(list(zip(cands.tolist(), fracs.tolist(), emit_logp.tolist())))

    # Viterbi
    T = len(pts)
    dp = [c[2] for c in candidates[0]]
    back = []  # back[t-1][j] = (best_i, intermediate_arcs)

    for t in range(1, T):
        d_euclid = math.hypot(pts[t, 0] - pts[t - 1, 0], pts[t, 1] - pts[t - 1, 1])
        max_route_dist = max(4.0 * d_euclid + 250.0, 500.0)

        prev_cands = candidates[t - 1]
        curr_cands = candidates[t]
        target_nodes = {int(arcs[a, 0]) for a, _, _ in curr_cands}

        # Cache Dijkstra per unique end-node of prev_cands
        dijk_cache = {}
        for a_i, _, _ in prev_cands:
            v_i = int(arcs[a_i, 1])
            if v_i not in dijk_cache:
                dijk_cache[v_i] = dijkstra_to_targets(adj, v_i, target_nodes, max_route_dist)

        new_dp = [-1e18] * len(curr_cands)
        step_back = [(0, []) for _ in range(len(curr_cands))]

        for i, (a_i, f_i, _) in enumerate(prev_cands):
            if dp[i] <= -1e17:
                continue
            v_i = int(arcs[a_i, 1])
            found, prev_map = dijk_cache[v_i]
            rem_i = (1.0 - f_i) * float(arc_lens[a_i])

            for j, (a_j, f_j, emit_j) in enumerate(curr_cands):
                u_j = int(arcs[a_j, 0])
                ent_j = f_j * float(arc_lens[a_j])

                if a_i == a_j and f_j >= f_i:
                    r_dist = (f_j - f_i) * float(arc_lens[a_i])
                    mid_arcs = []
                elif v_i == u_j:
                    r_dist = rem_i + ent_j
                    mid_arcs = []
                elif u_j in found:
                    r_dist = rem_i + found[u_j] + ent_j
                    mid_arcs = reconstruct_node_path(prev_map, v_i, u_j)
                else:
                    # Fallback penalty if disconnected
                    r_dist = d_euclid + 500.0
                    mid_arcs = []

                trans_logp = -abs(r_dist - d_euclid) / beta
                score = dp[i] + trans_logp + emit_j
                if score > new_dp[j]:
                    new_dp[j] = score
                    step_back[j] = (i, mid_arcs)

        # If all transitions failed, reset to emission-only from best prev
        if max(new_dp) <= -1e17:
            best_i = max(range(len(dp)), key=lambda k: dp[k])
            for j, (_, _, emit_j) in enumerate(curr_cands):
                new_dp[j] = dp[best_i] - 50.0 + emit_j
                step_back[j] = (best_i, [])

        dp = new_dp
        back.append(step_back)

    # Backtrack
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


def evaluate_route(pred: list, true: list, arc_lens: np.ndarray):
    pred_set = set(pred)
    true_set = set(true)
    inter = pred_set & true_set

    prec_cnt = len(inter) / len(pred_set) if pred_set else 0.0
    rec_cnt = len(inter) / len(true_set) if true_set else 0.0
    f1_cnt = 2 * prec_cnt * rec_cnt / (prec_cnt + rec_cnt) if (prec_cnt + rec_cnt) > 0 else 0.0

    len_inter = sum(float(arc_lens[a]) for a in inter)
    len_pred = sum(float(arc_lens[a]) for a in pred_set)
    len_true = sum(float(arc_lens[a]) for a in true_set)
    prec_len = len_inter / len_pred if len_pred > 0 else 0.0
    rec_len = len_inter / len_true if len_true > 0 else 0.0
    f1_len = 2 * prec_len * rec_len / (prec_len + rec_len) if (prec_len + rec_len) > 0 else 0.0

    return {
        "prec_cnt": prec_cnt,
        "rec_cnt": rec_cnt,
        "f1_cnt": f1_cnt,
        "prec_len": prec_len,
        "rec_len": rec_len,
        "f1_len": f1_len,
        "pred_n": len(pred),
        "true_n": len(true),
    }


def main():
    base = Path(__file__).resolve().parent / "Student folder" / "student_test"
    samples = sorted([d for d in base.iterdir() if d.is_dir()])

    print(f"Evaluating HMM baseline on {len(samples)} student_test samples...\n")
    print(f"{'Sample':<10} {'Pred':>6} {'True':>6} {'Cnt-F1':>8} {'Len-P':>8} {'Len-R':>8} {'Len-F1':>8} {'Time':>7}")
    print("-" * 70)

    metrics = []
    for s in samples:
        t0 = time.perf_counter()
        nodes, arcs, track, route = load_sample(s)
        nodes_xy, track_xy = project_to_meters(nodes, track[:, :2])
        pred_route, arc_lens = hmm_map_match(nodes_xy, arcs, track_xy)
        dt = time.perf_counter() - t0
        m = evaluate_route(pred_route, route, arc_lens)
        metrics.append(m)
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


if __name__ == "__main__":
    main()
