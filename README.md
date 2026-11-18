# ML-HACKATHON--Path-Snap
Map matching project that reconstructs road routes from GPS tracks using the supplied road network data. Includes route prediction code, evaluation tools, and generated predictions for the hackathon dataset.

## Model results

![Model performance by evaluation metric](reports/model_performance.svg)

The graph groups results by metric and test set. Compare bars within a group only; the percentages across groups measure different things. See [result sources and definitions](reports/README.md).

## Old and new dataset results

The two graphs below use **mean route length F1** for the same four HMM-family models. The old `student_test` set and new `evaluator_test` set each contain 10 different trips.

![Route length F1 on the new evaluator dataset](reports/new_dataset_route_f1.svg)

![Old student dataset versus new evaluator dataset](reports/old_vs_new_route_f1.svg)

The smoothed HMM scored 99.35% on `student_test` and 99.72% on `evaluator_test` (+0.37 percentage points). The hybrid also scored higher on the new set; the unsmoothed HMM and HGBC scored lower. These are scores on different trips, **not evidence that the models were retrained or improved**. KNN and MLP are absent because they do not have comparable route length F1 results on `evaluator_test`. See the [source scores and calculation](reports/README.md).
