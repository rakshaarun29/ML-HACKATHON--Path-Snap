import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

OUT_DIR = "artifacts"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# --- GRAPH 1: Model Benchmark Metrics Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

metrics_df = pd.DataFrame({
    "Metric": ["CV Macro F1", "Test Top-1 Acc", "Route Continuity", "Test Macro F1"],
    "Logistic Regression": [0.8144, 0.7904, 0.9147, 0.8607],
    "Random Forest": [0.8499, 0.8035, 0.9044, 0.8891]
})
x = np.arange(len(metrics_df))
w = 0.35
b1 = axes[0].bar(x - w/2, metrics_df["Logistic Regression"] * 100, w, label="Logistic Regression", color="#2b5c8f")
b2 = axes[0].bar(x + w/2, metrics_df["Random Forest"] * 100, w, label="Random Forest", color="#2ca02c")
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_df["Metric"], rotation=15)
axes[0].set_ylim(65, 100)
axes[0].set_ylabel("Score (%)")
axes[0].set_title("Classification & Route Continuity Metrics (Higher is Better)")
axes[0].legend()
for bars in (b1, b2):
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                     f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5)

loss_models = ["Logistic Regression", "Random Forest"]
loss_vals = [0.1307, 0.0943]
b3 = axes[1].bar(loss_models, loss_vals, width=0.45, color=["#d9534f", "#2ca02c"])
axes[1].set_ylim(0, 0.16)
axes[1].set_ylabel("Log Loss")
axes[1].set_title("Probability Calibration Error — Log Loss (Lower is Better)")
for bar in b3:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                 f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "model_benchmark_comparison.png"), dpi=150)
plt.close()

# --- Load test features & exported models for Graphs 2 & 3 ---
test_pkl = os.path.join(OUT_DIR, "test_features.pkl")
if os.path.exists(test_pkl):
    df_test = pd.read_pickle(test_pkl)
    df_test = df_test[df_test["label"] != -1].reset_index(drop=True)
    lr_bundle = joblib.load(os.path.join(OUT_DIR, "logistic_regression_model.joblib"))
    rf_bundle = joblib.load(os.path.join(OUT_DIR, "random_forest_model.joblib"))
    features = lr_bundle["features"]

    X_test = df_test[features].values.astype(np.float32)
    y_test = df_test["label"].values

    p_lr = lr_bundle["model"].predict_proba(X_test)[:, 1]
    p_rf = rf_bundle["model"].predict_proba(X_test)[:, 1]

    # --- GRAPH 2: Reliability (Calibration) Curve & Precision-Recall Curve ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, probs, color in [("Logistic Regression", p_lr, "#2b5c8f"), ("Random Forest", p_rf, "#2ca02c")]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=12)
        axes[0].plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name, color=color)
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Actual Positive Fraction")
    axes[0].set_title("Probability Calibration Curve (HMM Emission Readiness)")
    axes[0].legend()

    for name, probs, color in [("Logistic Regression", p_lr, "#2b5c8f"), ("Random Forest", p_rf, "#2ca02c")]:
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, linewidth=2, label=f"{name} (AP = {ap:.3f})", color=color)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Candidate Arc Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "calibration_and_pr_curves.png"), dpi=150)
    plt.close()

    # --- GRAPH 3: Confusion Matrices ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, probs) in zip(axes, [("Logistic Regression", p_lr), ("Random Forest", p_rf)]):
        cm = confusion_matrix(y_test, (probs >= 0.5).astype(int), normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Non-Match (0)", "True Arc (1)"],
                    yticklabels=["Non-Match (0)", "True Arc (1)"])
        ax.set_title(f"{name}\nNormalized Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

print("Generated all graphs in reports/figures/:")
for f in sorted(os.listdir(FIG_DIR)):
    print(" -", f)
