import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "perp_dist", "dist_rank", "dist_diff_min", "dist_ratio",
    "heading_diff", "cos_heading_diff", "speed_mps", "step_dist",
    "dt", "arc_length", "proj_pos_ratio", "u_in_deg", "v_out_deg",
    "is_same_arc", "is_direct_conn", "conn_gap_dist"
]

print("Loading training features...", flush=True)
df_train = pd.read_pickle(os.path.join(OUT_DIR, "train_features.pkl"))
df_train = df_train[df_train["label"] != -1].reset_index(drop=True)

X_train = df_train[FEATURE_COLS].values.astype(np.float32)
y_train = df_train["label"].values

print("Fitting & saving Logistic Regression...", flush=True)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
lr.fit(X_train, y_train)
lr_path = os.path.join(OUT_DIR, "logistic_regression_model.joblib")
joblib.dump({"model": lr, "features": FEATURE_COLS}, lr_path, compress=3)

print("Fitting & saving Random Forest...", flush=True)
rf = RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=30, max_samples=0.15, n_jobs=1, random_state=42)
rf.fit(X_train, y_train)
rf_path = os.path.join(OUT_DIR, "random_forest_model.joblib")
joblib.dump({"model": rf, "features": FEATURE_COLS}, rf_path, compress=3)

print(f"\nExported successfully:")
print(f" - {lr_path} ({os.path.getsize(lr_path) / 1024:.1f} KB)")
print(f" - {rf_path} ({os.path.getsize(rf_path) / (1024*1024):.2f} MB)")
