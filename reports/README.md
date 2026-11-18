# Model result sources

The [model performance graph](model_performance.svg) summarizes results already reported in the model branches. A higher bar is better **within its group**. These groups use different metrics and evaluation sets, so they do not establish one overall winning model.

| Graph group | Model | Reported score | Source |
| --- | --- | ---: | --- |
| GPS point-to-road match rate, 16 held-out `train` examples | KNN, k=20 | 63.85% | [`knn_results.json`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/knn-baseline/knn_results.json) |
| GPS point-to-road match rate, 16 held-out `train` examples | MLP | 78.81% | [`mlp_results.json`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/mlp-baseline/mlp_results.json) |
| Top-one arc accuracy, 10 `student_test` examples | Logistic regression | 79.04% | [`lr_rf_results.json`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/lr-rf-baseline/lr_rf_results.json) |
| Top-one arc accuracy, 10 `student_test` examples | Random forest | 80.35% | [`lr_rf_results.json`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/lr-rf-baseline/lr_rf_results.json) |
| Route length F1, 10 `student_test` examples | HMM | 98.84% | [`analysis_and_models.md`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/hmm-baseline/report/analysis_and_models.md) |
| Route length F1, 10 `student_test` examples | HMM + smoothing | 99.35% | [`analysis_and_models.md`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/hmm-baseline/report/analysis_and_models.md) |
| Route length F1, 10 `student_test` examples | HGBC + Viterbi | 97.07% | [`analysis_and_models.md`](https://github.com/rakshaarun29/ML-HACKATHON--Path-Snap/blob/hmm-baseline/report/analysis_and_models.md) |

KNN and MLP measure the fraction of individual GPS point matches whose arc appears anywhere in the reference route; they do not construct a connected route. Logistic regression and random forest report their `student_test_top1_acc_pct` field. The HMM figures are mean route length F1 scores. All scores come from the cited branches; this graph does not rerun or independently verify the models.

## Old and new dataset comparison

The [new dataset graph](new_dataset_route_f1.svg) and [old-versus-new graph](old_vs_new_route_f1.svg) use the mean of **per-trip route length F1** for the route-reconstruction methods. The source is [`hmm Model/report/results.txt`](../hmm%20Model/report/results.txt) for the 10 `student_test` trips and [`hmm Model/report/evaluator_results.txt`](../hmm%20Model/report/evaluator_results.txt) for the 10 different `evaluator_test` trips.

| Model | Old `student_test` | New `evaluator_test` | Change (percentage points) |
| --- | ---: | ---: | ---: |
| RF (Bayesian-Tuned) + Viterbi | 99.03% | 99.27% | +0.24 |
| Hybrid HMM + RF | 99.36% | 99.64% | +0.28 |
| HGBC + Viterbi | 97.07% | 96.12% | -0.95 |

The second graph zooms the horizontal axis to 95–100% so small differences are visible; the first uses a 0–100% scale. A higher score on the new set does not prove a model improved because the trips differ. These graphs only summarize scores already in the repository. They were not used to choose or tune a model. KNN and MLP are omitted because their saved results are point-match scores, not comparable route length F1 on `evaluator_test`.

To regenerate both SVGs from the saved score files: `pip install matplotlib`, then run `python reports/generate_dataset_graphs.py` from the repository root.
