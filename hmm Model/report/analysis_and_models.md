# Map-Matching Dataset Analysis & Model Guide

## 1. Dataset Overview
Location: `Student folder/` (90 samples total, 360 files)

- **`train/`**: 80 samples
- **`student_test/`**: 10 samples (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`)
- **Missing IDs (`00000000`–`00000099`)**: 10 IDs (`00000015`, `00000017`, `00000023`, `00000030`, `00000049`, `00000059`, `00000067`, `00000073`, `00000087`, `00000094`), likely reserved as a hidden evaluation split.

### File Schema
Each sample `<id>` contains 4 whitespace/tab-delimited files (0-indexed, no headers):

| File | Columns | Description |
| :--- | :--- | :--- |
| `<id>.nodes` | `lon  lat` | Road network vertices in decimal degrees. |
| `<id>.arcs` | `u  v` | Directed road segments from node `u` to node `v`. |
| `<id>.track` | `lon  lat  t` | Raw GPS trajectory points with timestamps `t` (seconds, nominally 1 Hz). |
| `<id>.route` | `arc_id` | Ground-truth sequence of arc indices from `<id>.arcs`. |

> **Note:** Coordinate order is **`lon lat`**, not `lat lon` (column 0 spans `-100.43` to `137.25`, column 1 spans `-34.65` to `61.28`).

### Summary Statistics

| Metric | `train` (80 samples) | `student_test` (10 samples) |
| :--- | :--- | :--- |
| **Nodes / sample** | min `299`, median `9,609`, max `155,003` | min `1,770`, median `6,745`, max `26,635` |
| **Arcs / sample** | min `649`, median `20,071`, max `307,392` | min `3,577`, median `13,184`, max `56,027` |
| **Track points / sample** | min `361`, median `1,551`, max `19,313` | min `262`, median `1,486`, max `3,958` |
| **Route arcs / sample** | min `59`, median `283`, max `2,266` | min `95`, median `268`, max `1,323` |
| **Track `dt` range** | `1.0s` to `153,064.0s` | `1.0s` to `210.0s` |

### Key Data Edge Cases
- **Directed graphs with parallel edges & self-loops**: 85.5%–99.6% of arcs in `student_test` have a reverse `(v, u)` arc; duplicate `(u, v)` pairs and occasional self-loops (`u == v`) exist.
- **Route connectivity gaps**: 10/80 `train` samples and 1/10 `student_test` sample (`00000076`) have 1–3 transitions in `.route` where `arcs[route[i]][1] != arcs[route[i+1]][0]`.
- **GPS noise & sampling gaps**: Raw `.track` distances exceed `.route` lengths (e.g., `00000077`: `36.29 km` track vs. `15.98 km` route), and `dt` has occasional signal gaps.

---

## 2. Benchmark Results (`student_test`)

| Model | Script | Mean Cnt-F1 | Mean Len-P | Mean Len-R | Mean Len-F1 | Time / sample |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HMM + Savitzky–Golay Smoothing** | `baseline_hmm.py` | **98.92%** | **99.09%** | **99.63%** | **99.35%** | **0.18s – 2.14s** |
| **Newson–Krumm HMM (unsmoothed)** | `baseline_hmm.py` | 97.78% | 98.15% | 99.55% | 98.84% | 0.06s – 1.38s |
| **HGBC (`sklearn`) + Viterbi** | `sklearn_hmm.py` | 95.89% | 94.92% | 99.47% | 97.07% | 1.69s – 24.84s |

### Per-Sample Accuracy (`baseline_hmm.py` with Savitzky–Golay Smoothing)

| Sample | Pred Arcs | True Arcs | Cnt-F1 | Len-P | Len-R | Len-F1 | Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 183 | 179 | 98.90% | 99.89% | 100.00% | 99.94% | 0.79s |
| `00000007` | 121 | 123 | 99.18% | 100.00% | 99.30% | 99.65% | 0.23s |
| `00000009` | 436 | 429 | 98.03% | 96.91% | 99.48% | 98.18% | 0.57s |
| `00000040` | 310 | 309 | 99.19% | 99.68% | 99.74% | 99.71% | 0.94s |
| `00000052` | 93 | 95 | 98.94% | 100.00% | 99.69% | 99.84% | 0.18s |
| `00000056` | 1322 | 1323 | 99.72% | 98.00% | 99.94% | 98.96% | 2.14s |
| `00000065` | 145 | 145 | 100.00% | 100.00% | 100.00% | 100.00% | 0.58s |
| `00000076` | 390 | 374 | 98.03% | 98.24% | 100.00% | 99.11% | 1.26s |
| `00000077` | 272 | 268 | 98.89% | 99.31% | 99.32% | 99.32% | 0.56s |
| `00000091` | 232 | 232 | 98.28% | 98.85% | 98.79% | 98.82% | 0.49s |
| **MEAN** | — | — | **98.92%** | **99.09%** | **99.63%** | **99.35%** | — |

### Per-Sample Accuracy (`baseline_hmm.py`)

| Sample | Pred Arcs | True Arcs | Cnt-F1 | Len-P | Len-R | Len-F1 | Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 189 | 179 | 96.20% | 99.60% | 99.82% | 99.71% | 0.55s |
| `00000007` | 123 | 123 | 98.78% | 98.92% | 99.30% | 99.11% | 0.16s |
| `00000009` | 445 | 429 | 96.79% | 95.95% | 99.14% | 97.52% | 0.50s |
| `00000040` | 319 | 309 | 98.71% | 98.71% | 99.74% | 99.22% | 0.46s |
| `00000052` | 93 | 95 | 98.94% | 100.00% | 99.69% | 99.84% | 0.06s |
| `00000056` | 1333 | 1323 | 99.16% | 96.62% | 100.00% | 98.28% | 1.38s |
| `00000065` | 149 | 145 | 100.00% | 100.00% | 100.00% | 100.00% | 0.44s |
| `00000076` | 407 | 374 | 94.10% | 95.67% | 99.08% | 97.35% | 0.69s |
| `00000077` | 329 | 268 | 97.28% | 97.81% | 100.00% | 98.89% | 0.26s |
| `00000091` | 235 | 232 | 97.85% | 98.20% | 98.79% | 98.49% | 0.32s |
| **MEAN** | — | — | **97.78%** | **98.15%** | **99.55%** | **98.84%** | — |

### Per-Sample Accuracy (`sklearn_hmm.py`)

| Sample | Pred Arcs | True Arcs | Cnt-F1 | Len-P | Len-R | Len-F1 | Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `00000000` | 202 | 179 | 93.44% | 98.19% | 99.91% | 99.04% | 15.51s |
| `00000007` | 127 | 123 | 97.19% | 97.77% | 99.30% | 98.53% | 6.36s |
| `00000009` | 473 | 429 | 95.38% | 92.55% | 98.51% | 95.44% | 4.26s |
| `00000040` | 323 | 309 | 99.04% | 96.91% | 100.00% | 98.43% | 11.63s |
| `00000052` | 94 | 95 | 98.41% | 99.27% | 99.69% | 99.48% | 1.69s |
| `00000056` | 1377 | 1323 | 97.62% | 95.93% | 100.00% | 97.92% | 24.84s |
| `00000065` | 153 | 145 | 97.64% | 96.82% | 100.00% | 98.38% | 7.05s |
| `00000076` | 429 | 374 | 91.14% | 91.14% | 98.02% | 94.45% | 8.55s |
| `00000077` | 395 | 268 | 90.54% | 81.95% | 100.00% | 90.08% | 6.12s |
| `00000091` | 233 | 232 | 98.49% | 98.69% | 99.28% | 98.99% | 12.99s |
| **MEAN** | — | — | **95.89%** | **94.92%** | **99.47%** | **97.07%** | — |

---

