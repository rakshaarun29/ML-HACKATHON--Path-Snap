import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("lr-rf/reports")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open("lr-rf/linreg_vs_rf_results.json", "r", encoding="utf-8") as f:
    reg_res = json.load(f)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

C_LIN = "#2B5B84"   # Steel Blue - Linear / Logistic Regression
C_RF  = "#D95F02"   # Burnt Orange - Random Forest

# ---------------------------------------------------------
# FIGURE 1: Route-Level Count-F1 & Length-F1 (Regression)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
splits = ["student_test", "evaluator_test"]
split_titles = ["Student Test (10 Trips)", "Evaluator Test (10 Trips)"]

for ax, sp, title in zip(axes, splits, split_titles):
    metrics = ["Route Count-F1", "Route Length-F1"]
    lin_vals = [
        reg_res["Linear Regression (OLS)"][sp]["route_cnt_f1_pct"],
        reg_res["Linear Regression (OLS)"][sp]["route_len_f1_pct"],
    ]
    rf_vals = [
        reg_res["Random Forest"][sp]["route_cnt_f1_pct"],
        reg_res["Random Forest"][sp]["route_len_f1_pct"],
    ]

    x = np.arange(len(metrics))
    w = 0.34
    b1 = ax.bar(x - w / 2, lin_vals, width=w, label="Linear Regression (OLS)", color=C_LIN, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, rf_vals, width=w, label="Random Forest Regressor", color=C_RF, edgecolor="black", linewidth=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_ylim(0, 100)
    if ax == axes[0]:
        ax.set_ylabel("F1 Score (%)", fontsize=10.5)

axes[1].legend(frameon=True, facecolor="white", loc="lower right", fontsize=9.5)
fig.suptitle("Route-Level Map-Matching Performance: Linear Regression vs. Random Forest", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_route_f1_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------
# FIGURE 2: Candidate-Level Fit & Discrimination (R2, AUC, Macro-F1, RMSE)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
cand_metrics = ["r2", "roc_auc", "cand_macro_f1", "rmse"]
cand_labels = ["R² (↑)", "ROC-AUC (↑)", "Macro-F1 (↑)", "RMSE (↓)"]

for ax, sp, title in zip(axes, splits, split_titles):
    lin_vals = [reg_res["Linear Regression (OLS)"][sp][m] for m in cand_metrics]
    rf_vals  = [reg_res["Random Forest"][sp][m] for m in cand_metrics]

    x = np.arange(len(cand_labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, lin_vals, width=w, label="Linear Regression (OLS)", color=C_LIN, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w / 2, rf_vals, width=w, label="Random Forest Regressor", color=C_RF, edgecolor="black", linewidth=0.6)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.018, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cand_labels, fontsize=10)
    ax.set_ylim(0, 0.98)
    if ax == axes[0]:
        ax.set_ylabel("Metric Value", fontsize=10.5)

axes[0].legend(frameon=True, facecolor="white", loc="upper left", fontsize=9.5)
fig.suptitle("Candidate-Level Calibration & Discrimination Metrics (Top-10 Arcs per GPS Ping)", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "02_candidate_metrics_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------
# FIGURE 3: Generalization Shift (student_test -> evaluator_test)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
panel_cfg = [
    ("route_len_f1_pct", "Route Length-F1 (%) — Higher is Better", (78, 88)),
    ("roc_auc", "Candidate ROC-AUC — Higher is Better", (0.74, 0.85)),
]

for ax, (key, title, ylim) in zip(axes, panel_cfg):
    lin_pts = [reg_res["Linear Regression (OLS)"]["student_test"][key], reg_res["Linear Regression (OLS)"]["evaluator_test"][key]]
    rf_pts  = [reg_res["Random Forest"]["student_test"][key], reg_res["Random Forest"]["evaluator_test"][key]]

    ax.plot(["student_test", "evaluator_test"], lin_pts, marker="o", linewidth=2.3, markersize=8, color=C_LIN, label="Linear Regression (OLS)")
    ax.plot(["student_test", "evaluator_test"], rf_pts, marker="s", linewidth=2.3, markersize=8, color=C_RF, label="Random Forest Regressor")

    for i, (lv, rv) in enumerate(zip(lin_pts, rf_pts)):
        fmt = "{:.2f}%" if "pct" in key else "{:.4f}"
        ax.annotate(fmt.format(lv), (i, lv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=C_LIN, fontweight="bold")
        ax.annotate(fmt.format(rv), (i, rv), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=9, color=C_RF, fontweight="bold")

    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_ylim(*ylim)
    ax.margins(x=0.25)

axes[0].legend(frameon=True, facecolor="white", loc="lower left", fontsize=9)
fig.suptitle("Cross-Split Generalization Stability (Student Test vs. Evaluator Test)", fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "03_generalization_stability.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------
# FIGURE 4: Regression vs. Classification Formulations on evaluator_test
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 4.5))
models_all = [
    "Logistic Regression\n(Classifier)",
    "Random Forest\n(Classifier)",
    "Linear Regression\n(OLS Regressor)",
    "Random Forest\n(Regressor)",
]
cnt_f1 = [64.87, 77.71, reg_res["Linear Regression (OLS)"]["evaluator_test"]["route_cnt_f1_pct"], reg_res["Random Forest"]["evaluator_test"]["route_cnt_f1_pct"]]
len_f1 = [72.23, 84.64, reg_res["Linear Regression (OLS)"]["evaluator_test"]["route_len_f1_pct"], reg_res["Random Forest"]["evaluator_test"]["route_len_f1_pct"]]

x = np.arange(len(models_all))
w = 0.35
b1 = ax.bar(x - w / 2, cnt_f1, width=w, label="Route Count-F1 (%)", color="#4E79A7", edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + w / 2, len_f1, width=w, label="Route Length-F1 (%)", color="#F28E2B", edgecolor="black", linewidth=0.6)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models_all, fontsize=10)
ax.set_ylim(0, 98)
ax.set_ylabel("F1 Score (%)", fontsize=10.5)
ax.set_title("Held-Out Evaluator Test (10 Trips): Regression vs. Classification Formulations", fontsize=12, fontweight="bold", pad=10)
ax.legend(frameon=True, facecolor="white", loc="upper left", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "04_formulation_comparison_evaluator.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------
# MARKDOWN REPORT (REPORT.md)
# ---------------------------------------------------------
report_md = """# Pointwise Map-Matching Evaluation: Linear Regression vs. Random Forest

## 1. Objective & Scope
This module evaluates **pointwise candidate-scoring models**—comparing linear models (**Linear Regression (OLS)** and **Logistic Regression**) against non-linear tree ensembles (**Random Forest Regressor** and **Random Forest Classifier**)—for GPS-to-road-network map matching without sequential post-processing.

---

## 2. Feature Engineering & Experimental Setup
* **Candidate Generation:** For every GPS observation, the top-$K$ ($K=10$) nearest road arcs are retrieved using vectorized equirectangular projection and chunked orthogonal segment projection (`chunk_size=256`).
* **16 Engineered GPS-to-Arc Features:**
  1. **Spatial Proximity (4):** Perpendicular distance (`perp_dist`), within-ping rank (`dist_rank`), gap to closest arc (`dist_diff_min`), and distance ratio (`dist_ratio`).
  2. **Kinematic & Directional Alignment (5):** Angular difference (`heading_diff`), cosine heading alignment (`cos_heading_diff`), instantaneous speed (`speed_mps`), step displacement (`step_dist`), and sampling interval (`dt`).
  3. **Road Geometry & Topology (4):** Arc length (`arc_length`), normalized projection position (`proj_pos_ratio`), start-node in-degree (`u_in_deg`), and end-node out-degree (`v_out_deg`).
  4. **Local Transition Continuity (3):** Same-arc indicator (`is_same_arc`), direct topological connection to previous Top-1 arc (`is_direct_conn`), and Euclidean connection gap (`conn_gap_dist`).
* **Dataset Splits:**
  * **Training:** 40 trajectories (`data/Student folder/train`)
  * **Validation (`student_test`):** 10 held-out trajectories
  * **Final Evaluation (`evaluator_test`):** 10 unseen trajectories

---

## 3. Quantitative Benchmark Results

### 3.1 Regression Formulation (`src/06_eval_linreg_vs_rf.py`)
Models predict continuous arc-membership scores $y \\in \\{0, 1\\}$, and each GPS ping selects the highest-scoring candidate arc ($\arg\max_a \hat{y}_a$), followed by consecutive duplicate compression.

| Split | Model | RMSE (↓) | $R^2$ (↑) | ROC-AUC (↑) | Candidate Macro-F1 (↑) | Route Count-F1 (↑) | Route Length-F1 (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`student_test` (10 trips)** | Linear Regression (OLS) | 0.4300 | 0.2246 | 0.7809 | 0.7216 | **77.86%** | 83.78% |
| **`student_test` (10 trips)** | **Random Forest Regressor** | **0.4024** | **0.3210** | **0.8149** | **0.7504** | 76.65% | **84.21%** |
| **`evaluator_test` (10 trips)** | Linear Regression (OLS) | 0.4319 | 0.2022 | 0.7699 | 0.6802 | **77.07%** | **83.98%** |
| **`evaluator_test` (10 trips)** | **Random Forest Regressor** | **0.4066** | **0.2929** | **0.8165** | **0.7114** | 75.29% | 82.86% |

### 3.2 Classification Formulation (`evaluator_test`, 10 Trips)

| Split | Model | Candidate Macro-F1 (↑) | Log Loss (↓) | Route Count-F1 (↑) | Route Length-F1 (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`evaluator_test` (10 trips)** | Logistic Regression | **0.5621** | **1.5484** | 64.87% | 72.23% |
| **`evaluator_test` (10 trips)** | **Random Forest Classifier** | 0.5541 | 1.7106 | **77.71%** | **84.64%** |

---

## 4. Visual Analysis

### 4.1 Route-Level Count-F1 & Length-F1
![Route F1 Comparison](figures/01_route_f1_comparison.png)

### 4.2 Candidate-Level Calibration & Discrimination
![Candidate Metrics Comparison](figures/02_candidate_metrics_comparison.png)

### 4.3 Cross-Split Generalization Stability
![Generalization Stability](figures/03_generalization_stability.png)

### 4.4 Regression vs. Classification Formulations (`evaluator_test`)
![Formulation Comparison](figures/04_formulation_comparison_evaluator.png)

---

## 5. Key Findings & Error Analysis
1. **Superior Candidate Scoring by Random Forest:** Across both held-out splits, **Random Forest Regressor** outperforms **Linear Regression (OLS)** on every candidate-level metric—cutting `RMSE` from `0.4319` to **`0.4066`**, raising $R^2$ from `0.2022` to **`0.2929`** (**+44.9% relative gain**), and increasing `ROC-AUC` from `0.7699` to **`0.8165`** on `evaluator_test`. Tree splits capture non-linear interactions between perpendicular distance (`perp_dist`), heading alignment (`cos_heading_diff`), and topological continuity (`is_direct_conn`).
2. **Classifier Gap (+12.41 pp Length-F1):** Under binary classification on `evaluator_test`, **Random Forest Classifier** (`84.64%` Length-F1, `77.71%` Count-F1) beats **Logistic Regression** (`72.23%` Length-F1, `64.87%` Count-F1) by **+12.41 percentage points**, confirming that a single linear hyperplane struggles with multi-lane and intersection geometry.
3. **Greedy Top-1 Tie-Breaking Behavior:** Under continuous OLS regression, **Linear Regression** slightly edges out **Random Forest Regressor** on `evaluator_test` route F1 (`83.98%` vs. `82.86%` Length-F1) because continuous linear combinations assign distinct real-valued scores to closely spaced parallel arcs, whereas shallow (`max_depth=12`) tree leaves can assign identical average leaf values to candidates falling in the same terminal node.
4. **Inherent Ceiling of Pointwise Models (~83%–85% Length-F1):** Evaluating each GPS ping independently without global path constraints causes occasional arc-hopping at dense urban intersections, bounding standalone pointwise accuracy around `84.6%` Route Length-F1.
"""

(OUT_DIR / "REPORT.md").write_text(report_md.strip() + "\n", encoding="utf-8")
print("Generated 4 figures in reports/linreg_vs_rf/figures/ and reports/linreg_vs_rf/REPORT.md")
