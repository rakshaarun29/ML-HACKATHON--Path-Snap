# MLP map matching baseline

This branch trains one global scikit-learn `MLPClassifier` to score nearby road arcs for each GPS point. Arc IDs differ between maps, so the model uses geometric features instead of treating arc IDs as classes. Features include point-to-segment distance, segment length, direction alignment, position along the segment, and distance rank.

A candidate arc gets label 1 when it appears in that example's known `.route`; other candidates get label 0. This is a weak point-level label because the data does not identify the exact arc for each GPS observation. The script balances the training rows by retaining up to four negative candidates per positive candidate.

## Run

Keep the dataset outside this public repository. From the repository folder in PowerShell:

```powershell
python -m pip install -r requirements.txt
python .\src\train_mlp.py "C:\Users\ruska\Downloads\Student folder\Student folder\train"
```

The script uses the same sorted 64/16 example split and the same point-match score as the KNN branch. It writes `mlp_results.json` and a local model file under `artifacts/`. The model file is excluded from Git by default.

The held-out score is the mean fraction of selected GPS point arcs that appear in the reference route. This baseline does not connect those arcs into a complete route and cannot be treated as the hackathon's full-route score.

## Recorded result

On the 16 held-out examples, the MLP point-match score was **78.8%** (mean across examples). The KNN baseline scored **63.9%** on the same split and metric, a difference of **14.9 percentage points**. The full per-example MLP results are in mlp_results.json.

