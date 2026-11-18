import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix
from lightgbm import LGBMClassifier

OUT_DIR = "artifacts"
FIG_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FEATURE_COLS = [
    "perp_dist", "dist_rank", "dist_diff_min", "dist_ratio",
    "heading_diff", "cos_heading_diff", "speed_mps", "step_dist",
    "dt", "arc_length", "proj_pos_ratio", "u_in_deg", "v_out_deg",
    "is_same_arc", "is_direct_conn", "conn_gap_dist"
]

print("Loading feature datasets...", flush=True)
df_train = pd.read_pickle(os.path.join(OUT_DIR, "train_features.pkl"))
df_test = pd.read_pickle(os.path.join(OUT_DIR, "test_features.pkl"))

df_train = df_train[df_train["label"] != -1].reset_index(drop=True)
df_test = df_test[df_test["label"] != -1].reset_index(drop=True)

X_train = df_train[FEATURE_COLS].values.astype(np.float32)
y_train = df_train["label"].values
groups_train = df_train["trip_id"].values

X_test = df_test[FEATURE_COLS].values.astype(np.float32)
y_test = df_test["label"].values
print(f"Loaded {len(X_train):,} train rows and {len(X_test):,} test rows.\n", flush=True)

def build_model(name, verbose_mlp=False):
    if name == "Logistic Regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=42))
    elif name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=42)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=40, max_depth=10, min_samples_leaf=30, max_samples=0.15, n_jobs=1, random_state=42)
    elif name == "LightGBM":
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=2, verbose=-1)
    elif name == "MLP Neural Net":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), batch_size=4096, max_iter=15, early_stopping=True, verbose=verbose_mlp, random_state=42))

MODEL_SPECS = [
    ("Logistic Regression", "Linear/Probabilistic", "C=1.0, StandardScaler"),
    ("Decision Tree", "Tree-based", "max_depth=12, min_samples_leaf=20"),
    ("Random Forest", "Ensemble/Bagging", "n_estimators=40, max_depth=10, max_samples=0.15"),
    ("LightGBM", "Boosting", "n_estimators=200, lr=0.05, num_leaves=31"),
    ("MLP Neural Net", "Neural Network", "hidden=(64,32), batch=4096, early_stopping=True"),
]

def evaluate_predictions(df_eval, probs):
    df_tmp = df_eval[["trip_id", "obs_idx", "arc_id", "u_node", "v_node", "label"]].copy()
    df_tmp["prob"] = probs

    preds_bin = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(df_tmp["label"], preds_bin, average="macro")
    prec = precision_score(df_tmp["label"], preds_bin, zero_division=0)
    rec = recall_score(df_tmp["label"], preds_bin, zero_division=0)
    ll = log_loss(df_tmp["label"], np.clip(probs, 1e-7, 1 - 1e-7))

    top1_idx = df_tmp.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    top1_df = df_tmp.loc[top1_idx].sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)
    top1_acc = (top1_df["label"] == 1).mean() * 100.0

    same_trip = top1_df["trip_id"].values[:-1] == top1_df["trip_id"].values[1:]
    same_arc = top1_df["arc_id"].values[:-1] == top1_df["arc_id"].values[1:]
    connected_arc = top1_df["v_node"].values[:-1] == top1_df["u_node"].values[1:]
    valid_transitions = (same_arc | connected_arc)[same_trip]
    route_continuity = valid_transitions.mean() * 100.0 if len(valid_transitions) > 0 else 0.0

    return {
        "top1_acc": top1_acc,
        "macro_f1": macro_f1,
        "precision": prec,
        "recall": rec,
        "log_loss": ll,
        "route_continuity": route_continuity
    }

gkf = GroupKFold(n_splits=5)
results = []
trained_models = {}

for m_idx, (name, family, config_str) in enumerate(MODEL_SPECS, start=1):
    print(f"[{m_idx}/{len(MODEL_SPECS)}] Training {name} ({family})...", flush=True)
    cv_f1s = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups_train), start=1):
        f_t0 = time.time()
        fold_model = build_model(name, verbose_mlp=False)
        fold_model.fit(X_train[tr_idx], y_train[tr_idx])
        va_probs = fold_model.predict_proba(X_train[va_idx])[:, 1]
        va_preds = (va_probs >= 0.5).astype(int)
        f1_val = f1_score(y_train[va_idx], va_preds, average="macro")
        cv_f1s.append(f1_val)
        print(f"    Fold {fold}/5 done in {time.time() - f_t0:.1f}s | Macro F1: {f1_val:.4f}", flush=True)

    cv_macro_f1 = float(np.mean(cv_f1s))

    print(f"    Fitting full model on all 80 training trips...", flush=True)
    full_model = build_model(name, verbose_mlp=True)
    t0 = time.time()
    full_model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    test_probs = full_model.predict_proba(X_test)[:, 1]
    inf_time = time.time() - t1

    metrics = evaluate_predictions(df_test, test_probs)
    trained_models[name] = full_model

    if name == "MLP Neural Net":
        mlp_step = full_model.named_steps["mlpclassifier"]
        plt.figure(figsize=(7, 4))
        plt.plot(mlp_step.loss_curve_, label="Training Loss")
        if hasattr(mlp_step, "validation_scores_"):
            plt.plot(mlp_step.validation_scores_, label="Validation Score")
        plt.title("MLP Neural Net Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss / Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "mlp_training_history.png"), dpi=150)
        plt.close()

    cm = confusion_matrix(y_test, (test_probs >= 0.5).astype(int))
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Match", "Match"], yticklabels=["Non-Match", "Match"])
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.close()

    row = {
        "Model": name,
        "Algorithm Family": family,
        "Key Configuration": config_str,
        "CV Macro F1": round(cv_macro_f1, 4),
        "Student-Test Top-1 Acc (%)": round(metrics["top1_acc"], 2),
        "Route Continuity (%)": round(metrics["route_continuity"], 2),
        "Student-Test Macro F1": round(metrics["macro_f1"], 4),
        "Precision": round(metrics["precision"], 4),
        "Recall": round(metrics["recall"], 4),
        "Log Loss": round(metrics["log_loss"], 4),
        "Train Time (s)": round(train_time, 2),
        "Inference Time (s)": round(inf_time, 4)
    }
    results.append(row)
    print(f"  => DONE {name} | CV F1: {row['CV Macro F1']} | Test Top-1: {row['Student-Test Top-1 Acc (%)']}% | Continuity: {row['Route Continuity (%)']}% | LogLoss: {row['Log Loss']} | Train: {row['Train Time (s)']}s\n", flush=True)

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
joblib.dump(trained_models, os.path.join(OUT_DIR, "baseline_models.joblib"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.barplot(data=df_res, x="Model", y="CV Macro F1", ax=axes[0], hue="Model", palette="viridis", legend=False)
axes[0].set_title("5-Fold Group CV Macro F1")
axes[0].tick_params(axis="x", rotation=25)
axes[0].set_ylim(0.5, 1.0)

sns.barplot(data=df_res, x="Model", y="Student-Test Top-1 Acc (%)", ax=axes[1], hue="Model", palette="magma", legend=False)
axes[1].set_title("Student-Test Top-1 Accuracy (%)")
axes[1].tick_params(axis="x", rotation=25)
axes[1].set_ylim(50, 100)

sns.barplot(data=df_res, x="Model", y="Log Loss", ax=axes[2], hue="Model", palette="coolwarm", legend=False)
axes[2].set_title("Student-Test Log Loss (Lower is Better)")
axes[2].tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_comparison_charts.png"), dpi=150)
plt.close()

print("\n=== CONSOLIDATED MODEL COMPARISON TABLE ===")
print(df_res.to_string(index=False))
