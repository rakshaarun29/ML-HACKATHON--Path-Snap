"""Compute and plot the arc-level confusion matrix on student_test."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent.parent
    test_dir = root.parent / "Student folder" / "student_test"
    pred_dir = root / "predictions"

    tp = fp = fn = tn = 0
    for s in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        sid = s.name
        with open(s / f"{sid}.arcs", "r", encoding="utf-8") as f:
            total_arcs = sum(1 for line in f if line.strip())
        with open(s / f"{sid}.route", "r", encoding="utf-8") as f:
            true_set = {int(line.strip()) for line in f if line.strip()}
        with open(pred_dir / f"{sid}.route", "r", encoding="utf-8") as f:
            pred_set = {int(line.strip()) for line in f if line.strip()}

        s_tp = len(pred_set & true_set)
        s_fp = len(pred_set - true_set)
        s_fn = len(true_set - pred_set)
        s_tn = total_arcs - (s_tp + s_fp + s_fn)

        tp += s_tp
        fp += s_fp
        fn += s_fn
        tn += s_tn

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums * 100.0

    prec = tp / (tp + fp) * 100.0
    rec = tp / (tp + fn) * 100.0
    f1 = 2 * prec * rec / (prec + rec)
    print(f"Confusion Matrix (HMM_Smoothed on 10 student_test graphs, {cm.sum():,} total arcs):")
    print(f"  TN={tn:,}  FP={fp:,}")
    print(f"  FN={fn:,}  TP={tp:,}")
    print(f"  Micro Precision={prec:.2f}%  Recall={rec:.2f}%  F1={f1:.2f}%")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    labels = ["Off Route (0)", "On Route (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(
        f"HMM_Smoothed Arc-Level Confusion Matrix (student_test)\n"
        f"Micro Precision={prec:.2f}% | Recall={rec:.2f}% | F1={f1:.2f}%",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )

    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            color = "white" if cm_pct[i, j] > 50 else "black"
            ax.text(
                j,
                i,
                f"{cell_names[i][j]}: {cm[i, j]:,}\n({cm_pct[i, j]:.2f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-Normalized (%)")
    plt.tight_layout()
    out_path = root / "report" / "confusion_matrix.png"
    plt.savefig(out_path, dpi=180)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
