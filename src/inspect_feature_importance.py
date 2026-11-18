import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = "artifacts"
FIG_DIR = "reports/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load both exported models
lr_bundle = joblib.load(os.path.join(OUT_DIR, "logistic_regression_model.joblib"))
rf_bundle = joblib.load(os.path.join(OUT_DIR, "random_forest_model.joblib"))

features = lr_bundle["features"]
lr_clf = lr_bundle["model"].named_steps["logisticregression"]
rf_clf = rf_bundle["model"]

# 2. Extract standardized coefficients & Gini importances
df_imp = pd.DataFrame({
    "Feature": features,
    "LR_Coef": lr_clf.coef_[0],
    "LR_Abs_Coef": np.abs(lr_clf.coef_[0]),
    "RF_Importance": rf_clf.feature_importances_
}).sort_values("RF_Importance", ascending=False).reset_index(drop=True)

print("\n=== FEATURE IMPORTANCE & COEFFICIENT COMPARISON ===")
print(df_imp.to_string(index=False))

# 3. Save dual horizontal bar chart for your presentation / README
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

df_lr_sorted = df_imp.sort_values("LR_Coef", ascending=True)
colors_lr = ["#d9534f" if c < 0 else "#2b5c8f" for c in df_lr_sorted["LR_Coef"]]
axes[0].barh(df_lr_sorted["Feature"], df_lr_sorted["LR_Coef"], color=colors_lr)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_title("Logistic Regression Standardized Coefficients\n(Blue = Increases Match Odds, Red = Decreases)")
axes[0].set_xlabel("Log-Odds Weight (Standardized)")

df_rf_sorted = df_imp.sort_values("RF_Importance", ascending=True)
sns.barplot(data=df_rf_sorted, x="RF_Importance", y="Feature", ax=axes[1], hue="Feature", palette="viridis", legend=False)
axes[1].set_title("Random Forest Feature Importance\n(Mean Decrease in Impurity)")
axes[1].set_xlabel("Gini Importance")

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "lr_rf_feature_importance.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved comparison plot to {fig_path}")
