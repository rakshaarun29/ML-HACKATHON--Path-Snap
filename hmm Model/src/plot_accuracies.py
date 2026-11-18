"""Plot per-sample and mean accuracies (Len-F1 and Cnt-F1) from a plain-text results file."""

import sys
from pathlib import Path
from collections import OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(txt_path: Path):
    """Parse whitespace-delimited `model_name sample_id cnt_f1 len_f1` rows from a .txt file."""
    data = OrderedDict()  # model -> {sample_id: (cnt_f1, len_f1)}
    samples = []
    seen_samples = set()

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            model, sid, cnt_f1, len_f1 = line.split()[:4]
            if model not in data:
                data[model] = {}
            data[model][sid] = (float(cnt_f1), float(len_f1))
            if sid not in seen_samples:
                seen_samples.add(sid)
                samples.append(sid)

    return data, samples


def main():
    default_txt = Path(__file__).resolve().parent.parent / "report" / "results.txt"
    txt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_txt
    data, samples = load_results(txt_path)

    labels = samples + ["MEAN"]
    models = list(data.keys())
    n_models = len(models)
    x = np.arange(len(labels))
    total_width = 0.8
    w = total_width / max(n_models, 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
    all_vals = []

    for idx, model in enumerate(models):
        offset = (idx - (n_models - 1) / 2.0) * w
        cnt_vals = [data[model].get(s, (0.0, 0.0))[0] for s in samples]
        len_vals = [data[model].get(s, (0.0, 0.0))[1] for s in samples]
        cnt_vals.append(float(np.mean(cnt_vals)))
        len_vals.append(float(np.mean(len_vals)))
        all_vals.extend(cnt_vals + len_vals)

        b1 = ax1.bar(x + offset, len_vals, w, label=model)
        b2 = ax2.bar(x + offset, cnt_vals, w, label=model)

        # Annotate MEAN bars
        for bar in (b1[-1], b2[-1]):
            h = bar.get_height()
            bar.axes.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.2,
                f"{h:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    y_min = max(0.0, math.floor(min(all_vals) - 2.0)) if "math" in globals() else max(0.0, float(np.floor(min(all_vals) - 2.0)))

    for ax, title, ylabel in [
        (ax1, "Length-Weighted F1 (Len-F1) on student_test", "Len-F1 (%)"),
        (ax2, "Arc-Count F1 (Cnt-F1) on student_test", "Cnt-F1 (%)"),
    ]:
        ax.set_ylim(y_min, 101.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axvline(len(labels) - 1.5, color="gray", linestyle=":", linewidth=1.2)

    ax1.legend(loc="lower left", ncol=min(n_models, 4), frameon=True)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax2.set_xlabel("student_test Sample ID", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else txt_path.parent / "accuracy_comparison.png"
    plt.savefig(out_path, dpi=180)
    print(f"Loaded {len(models)} models from {txt_path.name} -> saved {out_path.name}")


if __name__ == "__main__":
    main()
