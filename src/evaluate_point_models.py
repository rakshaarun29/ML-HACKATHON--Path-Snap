r"""Evaluate the existing KNN or MLP point-matching baseline on new journeys.

Run from the repository root after copying this file into src/:
    python .\src\evaluate_point_models.py knn "C:\path\to\evaluator_test"
    python .\src\evaluate_point_models.py mlp "C:\path\to\evaluator_test"

The .route file is used only for scoring. The reported point-match rate is
not the hackathon's connected-route metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate_knn(folder: Path) -> float:
    from knn_baseline import match_example

    predicted_arc_ids = match_example(folder, k=20)
    reference_route = np.loadtxt(
        folder / f"{folder.name}.route", dtype=int, ndmin=1
    )
    return float(np.isin(predicted_arc_ids, reference_route).mean())


def evaluate_mlp(folder: Path, bundle: dict) -> float:
    from train_mlp import candidate_features

    features, candidate_ids, labels = candidate_features(folder, k=bundle["k"])
    rows = features.reshape(-1, features.shape[-1])
    probabilities = bundle["model"].predict_proba(
        bundle["scaler"].transform(rows)
    )[:, 1]
    selected_column = probabilities.reshape(candidate_ids.shape).argmax(axis=1)
    selected_labels = labels[np.arange(len(candidate_ids)), selected_column]
    return float(selected_labels.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("knn", "mlp"))
    parser.add_argument("test_dir", type=Path)
    parser.add_argument("--model-file", type=Path, default=Path("artifacts/mlp_model.joblib"))
    args = parser.parse_args()

    if not args.test_dir.is_dir():
        parser.error(f"Test directory does not exist: {args.test_dir}")
    folders = sorted(path for path in args.test_dir.iterdir() if path.is_dir())
    if not folders:
        parser.error("No numbered journey folders found in the test directory")

    bundle = None
    if args.model == "mlp":
        if not args.model_file.is_file():
            parser.error(f"Saved MLP model not found: {args.model_file}")
        import joblib

        bundle = joblib.load(args.model_file)

    scores = {}
    for folder in folders:
        if args.model == "knn":
            score = evaluate_knn(folder)
        else:
            score = evaluate_mlp(folder, bundle)
        scores[folder.name] = score
        print(f"{folder.name}: {score:.1%}", flush=True)

    result = {
        "model": args.model.upper(),
        "metric": "mean fraction of GPS point arc matches present in reference route",
        "test_set": "evaluator_test",
        "example_count": len(scores),
        "per_example_scores": scores,
        "mean_score": float(np.mean(list(scores.values()))),
        "note": "This scores point matches, not completed connected routes.",
    }
    output = Path(f"{args.model}_evaluator_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Mean evaluator point-match rate: {result['mean_score']:.1%}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
