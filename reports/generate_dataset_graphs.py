"""Regenerate the two HMM-family dataset comparison charts from saved trip scores.

Run from anywhere with: python reports/generate_dataset_graphs.py
Requires matplotlib. This script reads results already committed to the repo;
it does not train or tune a model.
"""

from collections import defaultdict
import os
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "hmm Model" / "report"
DEST = ROOT / "reports"

MODELS = [
    ("HMM + smoothing", "HMM_Smoothed", "HMM_Smoothed"),
    ("Hybrid HMM + RF", "Hybrid_HMM_RF", "Hybrid_HMM_RF"),
    ("HMM, unsmoothed", "HMM_Baseline", "HMM_Unsmoothed"),
    ("HGBC + Viterbi", "sklearn_HGBC", "sklearn_HGBC"),
]
COLORS = ["#2563eb", "#0891b2", "#7c3aed", "#d97706"]


def read_scores(filename):
    scores = defaultdict(list)
    for raw in (SOURCE / filename).read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        model, trip, count_f1, length_f1 = raw.split()
        score = float(length_f1)
        if not 0 <= score <= 100:
            raise ValueError(f"Invalid route length F1: {filename} {trip} {score}")
        scores[model].append(score)
    return scores


old = read_scores("results.txt")
new = read_scores("evaluator_results.txt")
for label, old_key, new_key in MODELS:
    if len(old[old_key]) != 10 or len(new[new_key]) != 10:
        raise ValueError(f"Expected 10 trip scores in each set for {label}")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "svg.fonttype": "none",
})

labels = [row[0] for row in MODELS]
old_means = [mean(old[row[1]]) for row in MODELS]
new_means = [mean(new[row[2]]) for row in MODELS]


def save_svg(fig, filename):
    path = DEST / filename
    fig.savefig(path, facecolor=fig.get_facecolor())
    # Matplotlib leaves spaces after multiline path coordinates.
    content = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n",
                    encoding="utf-8")

# New set only: a zero-based scale avoids overstating small score differences.
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
bars = ax.barh(labels[::-1], new_means[::-1], color=COLORS[::-1], height=0.58)
ax.set_xlim(0, 110)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.xaxis.set_major_formatter(PercentFormatter(100))
ax.xaxis.grid(True, color="#dce4ee")
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0, pad=9, colors="#243247")
for bar, value in zip(bars, new_means[::-1]):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%", va="center", weight="bold", color="#172033")
ax.set_title("Route performance on the new evaluator set", loc="left",
             fontsize=18, weight="bold", color="#172033", pad=18)
ax.text(0, 1.02, "Mean route length F1 across 10 evaluator_test trips",
        transform=ax.transAxes, color="#526277", fontsize=11)
fig.text(0.075, 0.025,
         "Source: hmm Model/report/evaluator_results.txt  |  Route length F1 is not pointwise accuracy.",
         fontsize=9, color="#58687e")
fig.subplots_adjust(left=0.24, right=0.91, top=0.78, bottom=0.15)
save_svg(fig, "new_dataset_route_f1.svg")
if os.environ.get("PATHSNAP_PREVIEW_DIR"):
    fig.savefig(Path(os.environ["PATHSNAP_PREVIEW_DIR"]) / "new_dataset_route_f1.png",
                facecolor=fig.get_facecolor())
plt.close(fig)

# Old versus new: a paired chart, with a deliberately labelled zoomed axis.
fig, ax = plt.subplots(figsize=(10, 5.7), dpi=150)
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
for i, (label, before, after, color) in enumerate(zip(labels, old_means, new_means, COLORS)):
    y = 3 - i
    ax.plot([before, after], [y, y], color="#b9c6d8", lw=5, zorder=1)
    ax.scatter(before, y, s=145, facecolor="white", edgecolor=color,
               linewidth=2.5, zorder=3)
    ax.scatter(after, y, s=145, facecolor=color, edgecolor="white",
               linewidth=1, zorder=4)
    ax.text(95.15, y + 0.24, label, va="center", color="#172033", weight="bold")
    ax.text(100.16, y, f"{after-before:+.2f} pp", va="center", color=color,
            weight="bold", fontsize=11)
ax.set_xlim(95, 101.5)
ax.set_ylim(-0.65, 3.7)
ax.set_yticks([])
ax.set_xticks([95, 96, 97, 98, 99, 100])
ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.xaxis.grid(True, color="#dce4ee")
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0, pad=9, colors="#243247")
ax.set_xlabel("Mean route length F1 (zoomed axis: 95–100%)", labelpad=12,
              color="#334155")
ax.scatter([], [], s=110, facecolor="white", edgecolor="#64748b",
           linewidth=2, label="Old: student_test (10 trips)")
ax.scatter([], [], s=110, facecolor="#64748b", edgecolor="white",
           label="New: evaluator_test (10 different trips)")
ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2,
          frameon=False, fontsize=10)
ax.set_title("Old vs new: same models, different trips", loc="left",
             fontsize=18, weight="bold", color="#172033", pad=18)
ax.text(0, 1.02, "Change in mean route length F1; positive and negative changes are both shown",
        transform=ax.transAxes, color="#526277", fontsize=10)
fig.text(0.08, 0.035,
         "Scores come from separate datasets; a higher new score does not prove the model improved.",
         fontsize=9, color="#58687e")
fig.subplots_adjust(left=0.08, right=0.86, top=0.79, bottom=0.32)
save_svg(fig, "old_vs_new_route_f1.svg")
if os.environ.get("PATHSNAP_PREVIEW_DIR"):
    fig.savefig(Path(os.environ["PATHSNAP_PREVIEW_DIR"]) / "old_vs_new_route_f1.png",
                facecolor=fig.get_facecolor())
plt.close(fig)

for label, before, after in zip(labels, old_means, new_means):
    print(f"{label}: {before:.3f}% -> {after:.3f}% ({after-before:+.3f} pp)")
