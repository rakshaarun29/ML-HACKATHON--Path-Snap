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
