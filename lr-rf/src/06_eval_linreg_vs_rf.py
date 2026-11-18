import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, f1_score

import importlib.util
spec = importlib.util.spec_from_file_location("eval_mod", "src/05_eval_evaluator_test.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

DATA_ROOT = Path("data/Student folder")

def extract_16_features_chunked(tdir: Path, top_k: int = 10, chunk_size: int = 256) -> pd.DataFrame:
    trip_id = tdir.name
    track = pd.read_csv(tdir / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "ts"])
    nodes = pd.read_csv(tdir / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(tdir / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["u", "v"]).values
    route_f = tdir / f"{trip_id}.route"
    route_ids = pd.read_csv(route_f, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].values if route_f.exists() else None

    lat0 = np.deg2rad(nodes["lat"].mean())
    R = 6371000.0
    nx = (np.deg2rad(nodes["lon"].values) * R * np.cos(lat0)).astype(np.float32)
    ny = (np.deg2rad(nodes["lat"].values) * R).astype(np.float32)
    tx = (np.deg2rad(track["lon"].values) * R * np.cos(lat0)).astype(np.float32)
    ty = (np.deg2rad(track["lat"].values) * R).astype(np.float32)
    ts = track["ts"].values.astype(np.float64)

    ux, uy = nx[arcs[:, 0]], ny[arcs[:, 0]]
    vx, vy = nx[arcs[:, 1]], ny[arcs[:, 1]]
    dx, dy = vx - ux, vy - uy
    arc_len = np.hypot(dx, dy)
    arc_len_sq_safe = np.maximum(arc_len ** 2, 1e-12)
    arc_heading = np.arctan2(dy, dx)

    u_in_deg = np.bincount(arcs[:, 1], minlength=len(nodes))
    v_out_deg = np.bincount(arcs[:, 0], minlength=len(nodes))

    n_obs = len(track)
    step_dx = np.diff(tx, prepend=tx[0])
    step_dy = np.diff(ty, prepend=ty[0])
    step_dist = np.hypot(step_dx, step_dy)
    dt = np.maximum(np.diff(ts, prepend=ts[0]), 1.0).astype(np.float32)
    speed_mps = step_dist / dt
    gps_heading = np.arctan2(step_dy, step_dx)
    if n_obs > 1:
        gps_heading[0] = gps_heading[1]
        speed_mps[0] = speed_mps[1]

    k = min(top_k, len(arcs))
    cand_arcs = np.empty((n_obs, k), dtype=np.int64)
    cand_dists = np.empty((n_obs, k), dtype=np.float32)
    cand_t = np.empty((n_obs, k), dtype=np.float32)
    cand_px = np.empty((n_obs, k), dtype=np.float32)
    cand_py = np.empty((n_obs, k), dtype=np.float32)

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        px = tx[start:end, None] - ux[None, :]
        py = ty[start:end, None] - uy[None, :]
        t_proj = np.clip((px * dx[None, :] + py * dy[None, :]) / arc_len_sq_safe[None, :], 0.0, 1.0)
        proj_x = ux[None, :] + t_proj * dx[None, :]
        proj_y = uy[None, :] + t_proj * dy[None, :]
        dist_mat = np.hypot(tx[start:end, None] - proj_x, ty[start:end, None] - proj_y)

        part_idx = np.argpartition(dist_mat, kth=k - 1, axis=1)[:, :k]
        r_idx = np.arange(end - start)[:, None]
        part_dists = dist_mat[r_idx, part_idx]
        order = np.argsort(part_dists, axis=1)

        top_arcs = part_idx[r_idx, order]
        cand_arcs[start:end] = top_arcs
        cand_dists[start:end] = part_dists[r_idx, order]
        cand_t[start:end] = t_proj[r_idx, top_arcs]
        cand_px[start:end] = proj_x[r_idx, top_arcs]
        cand_py[start:end] = proj_y[r_idx, top_arcs]

    min_dist = cand_dists[:, [0]]
    dist_diff_min = cand_dists - min_dist
    dist_ratio = cand_dists / (min_dist + 1.0)
    dist_rank = np.broadcast_to(np.arange(k)[None, :], cand_arcs.shape)

    h_diff = np.abs((gps_heading[:, None] - arc_heading[cand_arcs] + np.pi) % (2 * np.pi) - np.pi)
    cos_h_diff = np.cos(h_diff)

    prev_top_arc = np.roll(cand_arcs[:, 0], 1)
    prev_top_arc[0] = -1
    prev_top_v = np.where(prev_top_arc >= 0, arcs[np.maximum(prev_top_arc, 0), 1], -1)
    prev_px = np.roll(cand_px[:, 0], 1)
    prev_py = np.roll(cand_py[:, 0], 1)

    u_nodes = arcs[cand_arcs, 0]
    v_nodes = arcs[cand_arcs, 1]
    is_same_arc = (cand_arcs == prev_top_arc[:, None]).astype(np.float32)
    is_direct_conn = (u_nodes == prev_top_v[:, None]).astype(np.float32)
    conn_gap = np.hypot(ux[u_nodes] - prev_px[:, None], uy[u_nodes] - prev_py[:, None])
    conn_gap[0, :] = 0.0
    conn_gap = np.where((is_same_arc == 1.0) | (is_direct_conn == 1.0), 0.0, conn_gap)

    route_set = set(route_ids.tolist()) if route_ids is not None else set()
    labels = np.isin(cand_arcs, list(route_set)).astype(np.int32)

    return pd.DataFrame({
        "trip_id": trip_id,
        "obs_idx": np.repeat(np.arange(n_obs), k),
        "arc_id": cand_arcs.ravel(),
        "label": labels.ravel(),
        "perp_dist": cand_dists.ravel(),
        "dist_rank": dist_rank.ravel(),
        "dist_diff_min": dist_diff_min.ravel(),
        "dist_ratio": dist_ratio.ravel(),
        "heading_diff": h_diff.ravel(),
        "cos_heading_diff": cos_h_diff.ravel(),
        "speed_mps": np.repeat(speed_mps, k),
        "step_dist": np.repeat(step_dist, k),
        "dt": np.repeat(dt, k),
        "arc_length": arc_len[cand_arcs].ravel(),
        "proj_pos_ratio": cand_t.ravel(),
        "u_in_deg": u_in_deg[u_nodes].ravel(),
        "v_out_deg": v_out_deg[v_nodes].ravel(),
        "is_same_arc": is_same_arc.ravel(),
        "is_direct_conn": is_direct_conn.ravel(),
        "conn_gap_dist": conn_gap.ravel(),
    })

def build_split_df(split_name: str, max_trips: int | None = None):
    split_dir = DATA_ROOT / split_name
    trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if max_trips:
        trip_dirs = trip_dirs[:max_trips]

    dfs, meta = [], {}
    for tdir in trip_dirs:
        nodes, arcs, track, route = eval_mod.load_sample(tdir)
        nodes_xy, track_xy = eval_mod.project_to_meters(nodes, track[:, :2])
        arc_lens = np.hypot(
            nodes_xy[arcs[:, 1], 0] - nodes_xy[arcs[:, 0], 0],
            nodes_xy[arcs[:, 1], 1] - nodes_xy[arcs[:, 0], 1],
        )
        meta[tdir.name] = (route, arc_lens)
        dfs.append(extract_16_features_chunked(tdir, top_k=10, chunk_size=256))
    return pd.concat(dfs, ignore_index=True), meta

def evaluate_regressor(model, df: pd.DataFrame, meta: dict, feat_cols: list[str]) -> dict:
    X = df[feat_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)
    scores = model.predict(X)
    preds = (scores >= 0.5).astype(int)

    df_tmp = df[["trip_id", "obs_idx", "arc_id"]].copy()
    df_tmp["score"] = scores
    top1 = (
        df_tmp.sort_values("score", ascending=False)
        .groupby(["trip_id", "obs_idx"], sort=False)
        .first()
        .reset_index()
        .sort_values(["trip_id", "obs_idx"])
    )

    route_metrics = []
    for tid, grp in top1.groupby("trip_id"):
        seq = grp["arc_id"].tolist()
        dedup_route = [seq[0]] + [a for i, a in enumerate(seq[1:], 1) if a != seq[i - 1]]
        true_route, arc_lens = meta[tid]
        route_metrics.append(eval_mod.evaluate_route(dedup_route, true_route, arc_lens))

    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y, scores))), 4),
        "r2": round(float(r2_score(y, scores)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "cand_macro_f1": round(float(f1_score(y.astype(int), preds, average="macro")), 4),
        "route_cnt_f1_pct": round(float(np.mean([m["f1_cnt"] for m in route_metrics])) * 100, 2),
        "route_len_f1_pct": round(float(np.mean([m["f1_len"] for m in route_metrics])) * 100, 2),
    }

def main():
    feat_cols = [
        "perp_dist", "dist_rank", "dist_diff_min", "dist_ratio",
        "heading_diff", "cos_heading_diff", "speed_mps", "step_dist",
        "dt", "arc_length", "proj_pos_ratio", "u_in_deg", "v_out_deg",
        "is_same_arc", "is_direct_conn", "conn_gap_dist",
    ]

    print("Extracting 16 features for train (40 trips), student_test, and evaluator_test...", flush=True)
    df_train, _ = build_split_df("train", max_trips=40)
    df_stu, meta_stu = build_split_df("student_test")
    df_eval, meta_eval = build_split_df("evaluator_test")

    X_tr = df_train[feat_cols].values.astype(np.float32)
    y_tr = df_train["label"].values.astype(np.float32)

    models = {
        "Linear Regression (OLS)": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, max_depth=12, min_samples_leaf=10, max_features="sqrt", n_jobs=-1, random_state=42
        ),
    }

    report = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(X_tr, y_tr)
        report[name] = {
            "student_test": evaluate_regressor(model, df_stu, meta_stu, feat_cols),
            "evaluator_test": evaluate_regressor(model, df_eval, meta_eval, feat_cols),
        }

    print("\n=== LINEAR REGRESSION VS RANDOM FOREST ===")
    for name, splits in report.items():
        print(f"\n{name}:")
        for split, metrics in splits.items():
            print(f"  {split:<15} -> {metrics}")

    with open("lr-rf/linreg_vs_rf_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved linreg_vs_rf_results.json")

if __name__ == "__main__":
    main()

