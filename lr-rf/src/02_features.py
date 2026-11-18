import os
import glob
import time
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

DATA_DIR = "data/Student folder"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

def get_utm_transformer(lon_mean, lat_mean):
    zone = int((lon_mean + 180) // 6) + 1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

def align_gps_to_route(track_xy, nodes_xy, arcs, route_arc_ids):
    T = len(track_xy)
    R = len(route_arc_ids)
    if R == 0:
        return np.full(T, -1, dtype=int)
    if R == 1:
        return np.full(T, route_arc_ids[0], dtype=int)

    r_arcs = arcs[route_arc_ids]
    x1, y1 = nodes_xy[r_arcs[:, 0], 0], nodes_xy[r_arcs[:, 0], 1]
    x2, y2 = nodes_xy[r_arcs[:, 1], 0], nodes_xy[r_arcs[:, 1], 1]

    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = np.where(dx * dx + dy * dy == 0, 1.0, dx * dx + dy * dy)

    px = track_xy[:, 0:1]
    py = track_xy[:, 1:2]
    proj_t = np.clip(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0)
    cost = np.hypot(px - (x1 + proj_t * dx), py - (y1 + proj_t * dy))

    dp = np.full((T, R), np.inf)
    back = np.zeros((T, R), dtype=int)
    dp[0, :] = cost[0, :] + np.arange(R) * 5.0

    for t in range(1, T):
        stay = dp[t - 1, :]
        step1 = np.concatenate([[np.inf], dp[t - 1, :-1]])
        step2 = np.concatenate([[np.inf, np.inf], dp[t - 1, :-2] + 10.0])
        choices = np.vstack([stay, step1, step2])
        best_choice = np.argmin(choices, axis=0)
        dp[t, :] = cost[t, :] + choices[best_choice, np.arange(R)]
        back[t, :] = np.arange(R) - best_choice

    seq_idx = np.zeros(T, dtype=int)
    seq_idx[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 2, -1, -1):
        seq_idx[t] = back[t + 1, seq_idx[t + 1]]

    return route_arc_ids[seq_idx]

def process_partition(partition_name, top_k=10):
    part_path = os.path.join(DATA_DIR, partition_name)
    trip_dirs = sorted([d for d in glob.glob(os.path.join(part_path, "*")) if os.path.isdir(d)])
    dfs = []
    miss_count = 0
    total_obs = 0

    for idx_t, tdir in enumerate(trip_dirs):
        trip_id = os.path.basename(tdir)
        track = pd.read_csv(os.path.join(tdir, f"{trip_id}.track"), sep=r"\s+", header=None, names=["lon", "lat", "ts"])
        nodes = pd.read_csv(os.path.join(tdir, f"{trip_id}.nodes"), sep=r"\s+", header=None, names=["lon", "lat"])
        arcs = pd.read_csv(os.path.join(tdir, f"{trip_id}.arcs"), sep=r"\s+", header=None, names=["u", "v"]).values
        route_f = os.path.join(tdir, f"{trip_id}.route")
        route_ids = pd.read_csv(route_f, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].values if os.path.exists(route_f) else None

        transformer = get_utm_transformer(track["lon"].mean(), track["lat"].mean())
        tx, ty = transformer.transform(track["lon"].values, track["lat"].values)
        nx_m, ny_m = transformer.transform(nodes["lon"].values, nodes["lat"].values)
        track_xy = np.column_stack([tx, ty])
        nodes_xy = np.column_stack([nx_m, ny_m])

        gt_arc_per_obs = align_gps_to_route(track_xy, nodes_xy, arcs, route_ids) if route_ids is not None else np.full(len(track), -1)

        x1, y1 = nodes_xy[arcs[:, 0], 0], nodes_xy[arcs[:, 0], 1]
        x2, y2 = nodes_xy[arcs[:, 1], 0], nodes_xy[arcs[:, 1], 1]
        arc_dx = x2 - x1
        arc_dy = y2 - y1
        arc_len = np.hypot(arc_dx, arc_dy)
        safe_len_sq = np.where(arc_len == 0, 1.0, arc_len * arc_len)
        arc_bearing = np.arctan2(arc_dy, arc_dx)

        # Build KDTree on arc midpoints, startpoints, and endpoints for fast spatial pre-filtering
        mid_xy = np.column_stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        start_xy = np.column_stack([x1, y1])
        end_xy = np.column_stack([x2, y2])
        all_pts = np.vstack([mid_xy, start_xy, end_xy])
        n_arcs = len(arcs)
        tree = cKDTree(all_pts)

        # Query 60 nearest spatial points per GPS observation in batch
        k_query = min(60, len(all_pts))
        _, nn_idx = tree.query(track_xy, k=k_query, workers=-1)
        if k_query == 1:
            nn_idx = nn_idx[:, None]
        nn_arc_idx = nn_idx % n_arcs  # map back to arc_id

        n_nodes = len(nodes)
        in_deg = np.bincount(arcs[:, 1], minlength=n_nodes)
        out_deg = np.bincount(arcs[:, 0], minlength=n_nodes)

        T = len(track)
        ts = track["ts"].values
        gps_dx = np.zeros(T)
        gps_dy = np.zeros(T)
        dt = np.ones(T)
        if T > 1:
            gps_dx[1:] = tx[1:] - tx[:-1]
            gps_dy[1:] = ty[1:] - ty[:-1]
            dt[1:] = np.maximum(ts[1:] - ts[:-1], 1e-3)
            gps_dx[0], gps_dy[0], dt[0] = gps_dx[1], gps_dy[1], dt[1]
        step_dist = np.hypot(gps_dx, gps_dy)
        speed = step_dist / dt
        gps_bearing = np.arctan2(gps_dy, gps_dx)

        k_actual = min(top_k, n_arcs)
        cand_ids_all = np.zeros((T, k_actual), dtype=int)
        cand_dists_all = np.zeros((T, k_actual), dtype=np.float64)
        cand_projs_all = np.zeros((T, k_actual), dtype=np.float64)

        for t in range(T):
            pool = np.unique(nn_arc_idx[t])
            if len(pool) < k_actual:
                pool = np.arange(n_arcs)
            proj_t = np.clip(((tx[t] - x1[pool]) * arc_dx[pool] + (ty[t] - y1[pool]) * arc_dy[pool]) / safe_len_sq[pool], 0.0, 1.0)
            dists = np.hypot(tx[t] - (x1[pool] + proj_t * arc_dx[pool]), ty[t] - (y1[pool] + proj_t * arc_dy[pool]))
            order = np.argsort(dists)[:k_actual]
            cand_ids_all[t] = pool[order]
            cand_dists_all[t] = dists[order]
            cand_projs_all[t] = proj_t[order]

        valid_gt = gt_arc_per_obs != -1
        if np.any(valid_gt):
            total_obs += int(np.sum(valid_gt))
            in_cands = np.any(cand_ids_all[valid_gt] == gt_arc_per_obs[valid_gt, None], axis=1)
            miss_count += int(np.sum(~in_cands))

        obs_idx = np.repeat(np.arange(T), k_actual)
        ranks = np.tile(np.arange(k_actual), T)
        flat_aids = cand_ids_all.ravel()
        flat_dists = cand_dists_all.ravel()
        flat_projs = cand_projs_all.ravel()
        min_dists = np.repeat(cand_dists_all[:, 0], k_actual)

        u_nodes = arcs[flat_aids, 0]
        v_nodes = arcs[flat_aids, 1]

        prev_top_aids = np.concatenate([[cand_ids_all[0, 0]], cand_ids_all[:-1, 0]])
        prev_top_v = np.repeat(arcs[prev_top_aids, 1], k_actual)
        prev_top_aid_rep = np.repeat(prev_top_aids, k_actual)

        is_same_arc = (flat_aids == prev_top_aid_rep).astype(np.float32)
        is_direct_conn = (u_nodes == prev_top_v).astype(np.float32)
        conn_gap_dist = np.hypot(nodes_xy[u_nodes, 0] - nodes_xy[prev_top_v, 0],
                                 nodes_xy[u_nodes, 1] - nodes_xy[prev_top_v, 1])
        conn_gap_dist = np.where((is_same_arc == 1.0) | (is_direct_conn == 1.0), 0.0, conn_gap_dist)

        ang_diff = np.abs((gps_bearing[obs_idx] - arc_bearing[flat_aids] + np.pi) % (2 * np.pi) - np.pi)
        ang_diff = np.where(step_dist[obs_idx] < 0.5, np.pi / 2.0, ang_diff)

        labels = np.where(gt_arc_per_obs[obs_idx] != -1, (flat_aids == gt_arc_per_obs[obs_idx]).astype(int), -1)

        dfs.append(pd.DataFrame({
            "trip_id": trip_id,
            "obs_idx": obs_idx,
            "ts": ts[obs_idx],
            "arc_id": flat_aids,
            "u_node": u_nodes,
            "v_node": v_nodes,
            "perp_dist": flat_dists,
            "dist_rank": ranks,
            "dist_diff_min": flat_dists - min_dists,
            "dist_ratio": (flat_dists + 0.1) / (min_dists + 0.1),
            "heading_diff": ang_diff,
            "cos_heading_diff": np.cos(ang_diff),
            "speed_mps": speed[obs_idx],
            "step_dist": step_dist[obs_idx],
            "dt": dt[obs_idx],
            "arc_length": arc_len[flat_aids],
            "proj_pos_ratio": flat_projs,
            "u_in_deg": in_deg[u_nodes],
            "v_out_deg": out_deg[v_nodes],
            "is_same_arc": is_same_arc,
            "is_direct_conn": is_direct_conn,
            "conn_gap_dist": conn_gap_dist,
            "label": labels
        }))
        if (idx_t + 1) % 20 == 0 or (idx_t + 1) == len(trip_dirs):
            print(f"  {partition_name}: {idx_t + 1}/{len(trip_dirs)} trips done...")

    df_feat = pd.concat(dfs, ignore_index=True)
    miss_rate = (miss_count / total_obs * 100) if total_obs > 0 else 0.0
    print(f"[{partition_name}] Processed {len(trip_dirs)} trips | {len(df_feat)} candidate rows | Candidate Miss Rate (K={top_k}): {miss_rate:.2f}% ({miss_count}/{total_obs})")
    return df_feat

t0 = time.time()
df_train_feat = process_partition("train", top_k=10)
df_test_feat = process_partition("student_test", top_k=10)

df_train_feat.to_pickle(os.path.join(OUT_DIR, "train_features.pkl"))
df_test_feat.to_pickle(os.path.join(OUT_DIR, "test_features.pkl"))
print(f"Saved feature datasets in {time.time() - t0:.1f}s")
