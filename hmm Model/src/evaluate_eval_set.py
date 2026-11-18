"""Evaluate HMM_Smoothed and saved sklearn_HGBC on evaluator_test and generate report figures."""

import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline_hmm import (
    load_sample,
    project_to_meters,
    hmm_map_match,
    evaluate_route,
    save_results_txt,
)
from sklearn_hmm import sklearn_hmm_match


def plot_eval_confusion_matrix(eval_dir: Path, pred_dir: Path, out_path: Path):
    """Compute and plot arc-level confusion matrix for HMM_Smoothed on evaluator_test."""
    tp = fp = fn = tn = 0
    for s in sorted(d for d in eval_dir.iterdir() if d.is_dir()):
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
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0
    prec = tp / (tp + fp) * 100.0
    rec = tp / (tp + fn) * 100.0
    f1 = 2 * prec * rec / (prec + rec)

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
        f"HMM_Smoothed Arc-Level Confusion Matrix (evaluator_test)\n"
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
                j, i,
                f"{cell_names[i][j]}: {cm[i, j]:,}\n({cm_pct[i, j]:.2f}%)",
                ha="center", va="center", color=color, fontsize=12, fontweight="bold",
            )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-Normalized (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    return tn, fp, fn, tp, prec, rec, f1


def plot_eval_accuracies(samples: list, hmm_rows: list, un_rows: list, sk_rows: list, out_path: Path):
    """Plot per-sample Len-F1 and Cnt-F1 comparison on evaluator_test."""
    labels = samples + ["MEAN"]
    x = np.arange(len(labels))
    w = 0.26

    hmm_cnt = [r[1] * 100 for r in hmm_rows]
    hmm_len = [r[2] * 100 for r in hmm_rows]
    hmm_cnt.append(float(np.mean(hmm_cnt)))
    hmm_len.append(float(np.mean(hmm_len)))

    un_cnt = [r[1] * 100 for r in un_rows]
    un_len = [r[2] * 100 for r in un_rows]
    un_cnt.append(float(np.mean(un_cnt)))
    un_len.append(float(np.mean(un_len)))

    sk_cnt = [r[1] * 100 for r in sk_rows]
    sk_len = [r[2] * 100 for r in sk_rows]
    sk_cnt.append(float(np.mean(sk_cnt)))
    sk_len.append(float(np.mean(sk_len)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    for ax, h_vals, u_vals, s_vals, title, ylabel in [
        (ax1, hmm_len, un_len, sk_len, "Length-Weighted F1 (Len-F1) on evaluator_test", "Len-F1 (%)"),
        (ax2, hmm_cnt, un_cnt, sk_cnt, "Arc-Count F1 (Cnt-F1) on evaluator_test", "Cnt-F1 (%)"),
    ]:
        b1 = ax.bar(x - w, h_vals, w, label="HMM_Smoothed", color="#2ecc71")
        b2 = ax.bar(x,     u_vals, w, label="HMM_Unsmoothed", color="#3498db")
        b3 = ax.bar(x + w, s_vals, w, label="sklearn_HGBC", color="#e67e22")
        y_min = max(0.0, float(np.floor(min(h_vals + u_vals + s_vals) - 3.0)))
        ax.set_ylim(y_min, 102.0)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axvline(len(labels) - 1.5, color="gray", linestyle=":", linewidth=1.2)
        for bar in (b1[-1], b2[-1], b3[-1]):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.legend(loc="lower left", ncol=3, frameon=True)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax2.set_xlabel("evaluator_test Sample ID", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parent.parent
    data_root = root.parent
    eval_dir = data_root / "evaluator_test" / "evaluator_test"
    if not eval_dir.exists():
        eval_dir = data_root / "evaluator_test"
    samples = sorted([d for d in eval_dir.iterdir() if d.is_dir()])

    pred_dir = root / "predictions" / "evaluator_test"
    pred_dir.mkdir(parents=True, exist_ok=True)
    report_dir = root / "report"
    report_dir.mkdir(exist_ok=True)

    hmm_rows = []
    hmm_details = []
    for s in samples:
        t0 = time.perf_counter()
        n, a, t, r = load_sample(s)
        xy_n, xy_t = project_to_meters(n, t[:, :2])
        pred, alens = hmm_map_match(xy_n, a, xy_t, timestamps=t[:, 2])
        dt = time.perf_counter() - t0
        (pred_dir / f"{s.name}.route").write_text("\n".join(map(str, pred)) + "\n", encoding="utf-8")
        m = evaluate_route(pred, r, alens)
        hmm_rows.append((s.name, m["f1_cnt"], m["f1_len"]))
        hmm_details.append((s.name, m, dt))

    un_rows = []
    un_details = []
    for s in samples:
        t0 = time.perf_counter()
        n, a, t, r = load_sample(s)
        xy_n, xy_t = project_to_meters(n, t[:, :2])
        pred, alens = hmm_map_match(xy_n, a, xy_t, timestamps=t[:, 2], use_smoothing=False)
        dt = time.perf_counter() - t0
        m = evaluate_route(pred, r, alens)
        un_rows.append((s.name, m["f1_cnt"], m["f1_len"]))
        un_details.append((s.name, m, dt))

    bundle = joblib.load(root / "models" / "sklearn_hgbc_hmm.joblib")
    emit_clf, trans_clf = bundle["emit_clf"], bundle["trans_clf"]
    sk_rows = []
    sk_details = []
    for s in samples:
        t0 = time.perf_counter()
        n, a, t, r = load_sample(s)
        xy_n, xy_t = project_to_meters(n, t[:, :2])
        pred, alens = sklearn_hmm_match(xy_n, a, xy_t, emit_clf, trans_clf)
        dt = time.perf_counter() - t0
        m = evaluate_route(pred, r, alens)
        sk_rows.append((s.name, m["f1_cnt"], m["f1_len"]))
        sk_details.append((s.name, m, dt))

    eval_txt = report_dir / "evaluator_results.txt"
    save_results_txt("HMM_Smoothed", hmm_rows, eval_txt)
    save_results_txt("HMM_Unsmoothed", un_rows, eval_txt)
    save_results_txt("sklearn_HGBC", sk_rows, eval_txt)

    acc_png = report_dir / "evaluator_accuracy_comparison.png"
    plot_eval_accuracies([s.name for s in samples], hmm_rows, un_rows, sk_rows, acc_png)

    cm_png = report_dir / "evaluator_confusion_matrix.png"
    tn, fp, fn, tp, prec, rec, f1 = plot_eval_confusion_matrix(eval_dir, pred_dir, cm_png)

    print(
        f"HMM_Unsmoothed MEAN: Cnt-F1={np.mean([m['f1_cnt'] for _, m, _ in un_details])*100:.2f}% "
        f"Len-P={np.mean([m['prec_len'] for _, m, _ in un_details])*100:.2f}% "
        f"Len-R={np.mean([m['rec_len'] for _, m, _ in un_details])*100:.2f}% "
        f"Len-F1={np.mean([m['f1_len'] for _, m, _ in un_details])*100:.2f}%"
    )
    for sid, m, _ in un_details:
        print(f"  {sid}: pred={m['pred_n']} true={m['true_n']} cnt={m['f1_cnt']*100:.2f}% len={m['f1_len']*100:.2f}%")
    print(f"Saved {eval_txt}, {acc_png}, {cm_png}")
    print(f"CM: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,} | Micro F1={f1:.2f}%")


if __name__ == "__main__":
    main()
