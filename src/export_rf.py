import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

OUT_DIR = "artifacts"
FEATURE_COLS = [
    "perp_dist", "dist_rank", "dist_diff_min", "dist_ratio",
    "heading_diff", "cos_heading_diff", "speed_mps", "step_dist",
    "dt", "arc_length", "proj_pos_ratio", "u_in_deg", "v_out_deg",
    "is_same_arc", "is_direct_conn", "conn_gap_dist"
]

df_train = pd.read_pickle(os.path.join(OUT_DIR, "train_features.pkl"))
df_train = df_train[df_train["label"] != -1].reset_index(drop=True)

rf = RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=30, max_samples=0.15, n_jobs=1, random_state=42)
rf.fit(df_train[FEATURE_COLS].values.astype(np.float32), df_train["label"].values)

joblib.dump({"model": rf, "features": FEATURE_COLS}, os.path.join(OUT_DIR, "random_forest_model.joblib"), compress=3)
print("Saved artifacts/random_forest_model.joblib successfully!")
