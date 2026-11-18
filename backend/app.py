import json
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Path-Snap Map-Matching API", version="1.0.0")

# Allow your friend's frontend laptop to call this API over Wi-Fi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent
LR_RF_DIR = ROOT / "lr-rf"
HMM_DIR = ROOT / "hmm Model"
DATA_DIR = ROOT / "data" / "Student folder"

# Serve generated charts directly as static images for the frontend
if (LR_RF_DIR / "reports" / "figures").exists():
    app.mount("/figures/lr-rf", StaticFiles(directory=str(LR_RF_DIR / "reports" / "figures")), name="lr_rf_figures")
if (HMM_DIR / "report").exists():
    app.mount("/figures/hmm", StaticFiles(directory=str(HMM_DIR / "report")), name="hmm_figures")

# Load trained LR/RF model bundles (unwrap dict if saved with metadata)
def _load_bundle(path: Path):
    obj = joblib.load(path)
    if isinstance(obj, dict):
        est = obj.get("model") or obj.get("estimator") or obj.get("clf") or next(
            (v for v in obj.values() if hasattr(v, "predict")), None
        )
        cols = obj.get("feature_cols") or obj.get("features")
        return est, cols
    return obj, None

rf_model, rf_cols = _load_bundle(LR_RF_DIR / "artifacts" / "random_forest_model.joblib")
lr_model, lr_cols = _load_bundle(LR_RF_DIR / "artifacts" / "logistic_regression_model.joblib")


def compress_consecutive(seq):
    out = []
    for x in seq:
        x = int(x)
        if not out or out[-1] != x:
            out.append(x)
    return out


def extract_features_for_trip(track_df: pd.DataFrame, nodes_df: pd.DataFrame, arcs_arr: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    lat0 = np.deg2rad(nodes_df["lat"].mean())
    R = 6371000.0
    nx = (np.deg2rad(nodes_df["lon"].values) * R * np.cos(lat0)).astype(np.float32)
    ny = (np.deg2rad(nodes_df["lat"].values) * R).astype(np.float32)
    tx = (np.deg2rad(track_df["lon"].values) * R * np.cos(lat0)).astype(np.float32)
    ty = (np.deg2rad(track_df["lat"].values) * R).astype(np.float32)
    ts = track_df["ts"].values.astype(np.float64)

    ux, uy = nx[arcs_arr[:, 0]], ny[arcs_arr[:, 0]]
    vx, vy = nx[arcs_arr[:, 1]], ny[arcs_arr[:, 1]]
    dx, dy = vx - ux, vy - uy
    arc_len = np.hypot(dx, dy)
    arc_len_sq_safe = np.maximum(arc_len ** 2, 1e-12)
    arc_heading = np.arctan2(dy, dx)

    u_in_deg = np.bincount(arcs_arr[:, 1], minlength=len(nodes_df))
    v_out_deg = np.bincount(arcs_arr[:, 0], minlength=len(nodes_df))

    n_obs = len(track_df)
    step_dx = np.diff(tx, prepend=tx[0])
    step_dy = np.diff(ty, prepend=ty[0])
    dt = np.maximum(np.diff(ts, prepend=ts[0]), 1e-3)
    gps_speed = np.hypot(step_dx, step_dy) / dt
    gps_heading = np.arctan2(step_dy, step_dx)
    gps_speed[0] = gps_speed[1] if n_obs > 1 else 0.0
    gps_heading[0] = gps_heading[1] if n_obs > 1 else 0.0

    rows = []
    k = min(top_k, len(arcs_arr))
    for i in range(n_obs):
        px, py = tx[i], ty[i]
        t_proj = np.clip(((px - ux) * dx + (py - uy) * dy) / arc_len_sq_safe, 0.0, 1.0)
        proj_x = ux + t_proj * dx
        proj_y = uy + t_proj * dy
        dist = np.hypot(px - proj_x, py - proj_y)

        cand_idx = np.argpartition(dist, k - 1)[:k] if k < len(arcs_arr) else np.arange(len(arcs_arr))
        cand_idx = cand_idx[np.argsort(dist[cand_idx])]
        d_c = dist[cand_idx]
        t_c = t_proj[cand_idx]
        h_diff = np.abs((gps_heading[i] - arc_heading[cand_idx] + np.pi) % (2 * np.pi) - np.pi)

        for rank, a_id in enumerate(cand_idx):
            rows.append({
                "obs_idx": i,
                "arc_id": int(a_id),
                "dist": float(d_c[rank]),
                "dist_sq": float(d_c[rank] ** 2),
                "dist_rank": rank,
                "dist_ratio_min": float(d_c[rank] / max(d_c[0], 1e-3)),
                "dist_diff_min": float(d_c[rank] - d_c[0]),
                "t_proj": float(t_c[rank]),
                "near_endpoint": float(min(t_c[rank], 1.0 - t_c[rank])),
                "arc_len": float(arc_len[a_id]),
                "cos_heading_diff": float(np.cos(h_diff[rank])),
                "sin_heading_diff": float(np.sin(h_diff[rank])),
                "abs_heading_diff": float(h_diff[rank]),
                "gps_speed": float(gps_speed[i]),
                "dt": float(dt[i]),
                "u_in_deg": int(u_in_deg[arcs_arr[a_id, 0]]),
                "v_out_deg": int(v_out_deg[arcs_arr[a_id, 1]]),
                "is_oneway": 1,
            })
    return pd.DataFrame(rows)


def run_inference(track_df: pd.DataFrame, nodes_df: pd.DataFrame, arcs_arr: np.ndarray, model_name: str, trip_id: str = "custom"):
    if model_name == "hmm":
        pred_file = HMM_DIR / "predictions" / f"{trip_id}.route"
        if pred_file.exists():
            snapped_arcs = pd.read_csv(pred_file, sep=r"\s+", header=None, names=["arc_id"])["arc_id"].tolist()
            snapped_arcs = compress_consecutive(snapped_arcs)
        else:
            raise HTTPException(status_code=404, detail=f"No precomputed HMM route for trip '{trip_id}'. Use 'random_forest' or 'logistic_regression'.")
    else:
        feat_df = extract_features_for_trip(track_df, nodes_df, arcs_arr)
        model, saved_cols = (rf_model, rf_cols) if model_name == "random_forest" else (lr_model, lr_cols)
        feature_cols = saved_cols if saved_cols else [c for c in feat_df.columns if c not in ("obs_idx", "arc_id")]
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        if hasattr(model, "predict_proba"):
            feat_df["score"] = model.predict_proba(feat_df[feature_cols])[:, 1]
        else:
            feat_df["score"] = model.predict(feat_df[feature_cols])
        best_per_obs = feat_df.loc[feat_df.groupby("obs_idx")["score"].idxmax()].sort_values("obs_idx")
        snapped_arcs = compress_consecutive(best_per_obs["arc_id"].tolist())

    # Convert snapped arc IDs into [lon, lat] polylines for Leaflet/Mapbox rendering
    snapped_coords = []
    for a_id in snapped_arcs:
        if 0 <= a_id < len(arcs_arr):
            u, v = arcs_arr[a_id]
            snapped_coords.append([float(nodes_df.iloc[u]["lon"]), float(nodes_df.iloc[u]["lat"])])
            snapped_coords.append([float(nodes_df.iloc[v]["lon"]), float(nodes_df.iloc[v]["lat"])])

    raw_pings = track_df[["lon", "lat"]].values.tolist()
    return {
        "trip_id": trip_id,
        "model": model_name,
        "num_raw_pings": len(raw_pings),
        "num_snapped_arcs": len(snapped_arcs),
        "raw_pings": raw_pings,
        "snapped_arcs": snapped_arcs,
        "snapped_coords": snapped_coords,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models_available": ["random_forest", "logistic_regression", "hmm"],
        "local_data_available": DATA_DIR.exists(),
    }


@app.get("/api/metrics")
def get_metrics():
    results_path = LR_RF_DIR / "linreg_vs_rf_results.json"
    with open(results_path, "r") as f:
        lr_rf_metrics = json.load(f)
    return {
        "lr_rf": lr_rf_metrics,
        "hmm_summary": {
            "baseline_hmm_acc": 0.884,
            "hybrid_hmm_rf_acc": 0.912,
        },
        "figures": {
            "route_f1": "/figures/lr-rf/01_route_f1_comparison.png",
            "candidate_metrics": "/figures/lr-rf/02_candidate_metrics_comparison.png",
            "generalization": "/figures/lr-rf/03_generalization_stability.png",
            "formulation": "/figures/lr-rf/04_formulation_comparison_evaluator.png",
            "hmm_accuracy": "/figures/hmm/evaluator_accuracy_comparison.png",
        },
    }


@app.get("/api/trips")
def list_trips():
    trips = set()
    if DATA_DIR.exists():
        for split in ("student_test", "evaluator_test", "train"):
            sdir = DATA_DIR / split
            if sdir.exists():
                trips.update([p.name for p in sdir.iterdir() if p.is_dir()])
    if (HMM_DIR / "predictions").exists():
        trips.update([p.stem for p in (HMM_DIR / "predictions").glob("*.route")])
    return {"trips": sorted(trips)}


@app.get("/api/snap/{trip_id}")
def snap_existing_trip(trip_id: str, model: str = "random_forest"):
    for split in ("student_test", "evaluator_test", "train"):
        tdir = DATA_DIR / split / trip_id
        if tdir.exists():
            track = pd.read_csv(tdir / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "ts"])
            nodes = pd.read_csv(tdir / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"])
            arcs = pd.read_csv(tdir / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["u", "v"]).values
            return run_inference(track, nodes, arcs, model, trip_id=trip_id)
    raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in data/Student folder.")


@app.post("/api/snap/upload")
async def snap_uploaded_trip(
    model: str = "random_forest",
    track_file: UploadFile = File(...),
    nodes_file: UploadFile = File(...),
    arcs_file: UploadFile = File(...),
):
    track = pd.read_csv(track_file.file, sep=r"\s+", header=None, names=["lon", "lat", "ts"])
    nodes = pd.read_csv(nodes_file.file, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(arcs_file.file, sep=r"\s+", header=None, names=["u", "v"]).values
    return run_inference(track, nodes, arcs, model, trip_id="uploaded_trip")

