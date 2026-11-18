"""Choose k for the KNN map-matching baseline using labeled train examples."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from knn_baseline import match_example


def score(example: Path, k: int) -> float:
    predicted = match_example(example, k)
    route = np.loadtxt(example / f"{example.name}.route", dtype=int, ndmin=1)
    return float(np.isin(predicted, route).mean())


def mean_score(examples: list[Path], k: int) -> float:
    values = []
    for number, example in enumerate(examples, 1):
        value = score(example, k)
        values.append(value)
        print(f"  {number}/{len(examples)} {example.name}: {value:.1%}", flush=True)
    return float(np.mean(values))


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", type=Path, help="Path to Student folder/train")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--output", type=Path, default=Path("knn_results.json"))
    parser.add_argument("--limit", type=int, help="Use this many examples for a quick first run")
    args = parser.parse_args()

    if any(k < 1 for k in args.k):
        parser.error("Every k must be at least 1")
    examples = sorted(path for path in args.train_dir.iterdir() if path.is_dir())
    if args.limit is not None:
        if args.limit < 2:
            parser.error("--limit must be at least 2")
        examples = examples[: args.limit]
    if len(examples) < 2:
        parser.error("Need at least two training examples")

    split = max(1, int(len(examples) * 0.8))
    tuning, validation = examples[:split], examples[split:]
    print(f"Tuning on {len(tuning)} examples; validating on {len(validation)}")
    tuning_scores = {}
    for k in args.k:
        print(f"k={k}")
        tuning_scores[str(k)] = mean_score(tuning, k)
        print(f"Mean tuning score: {tuning_scores[str(k)]:.1%}")

    best_k = max(args.k, key=lambda k: tuning_scores[str(k)])
    print(f"Validating selected k={best_k}")
    validation_score = mean_score(validation, best_k)
    result = {
        "method": "nearest road segment KNN baseline",
        "score_definition": "fraction of GPS point arc matches present in reference route",
        "example_count": len(examples),
        "tuning_count": len(tuning),
        "validation_count": len(validation),
        "tuning_scores": tuning_scores,
        "best_k": best_k,
        "validation_score": validation_score,
        "note": "Point matches are not yet a connected route; this is not the hackathon metric.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {args.output}")
    print(f"Held-out validation score: {validation_score:.1%}")


if __name__ == "__main__":
    main()
