"""Train an MLP to rank nearby road arcs for each GPS observation.

Run from the repository root:
    python src/train_mlp.py "C:\path\to\Student folder\train"

The model uses geometric features, since arc IDs differ between maps. The label
for a candidate is whether its arc appears in the reference route. This is a
point-matching baseline, not a complete connected-route predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def read_example(folder: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name = folder.name
    nodes = np.loadtxt(folder / f"{name}.nodes", dtype=float, ndmin=2)[:, :2]
    arcs = np.loadtxt(folder / f"{name}.arcs", dtype=int, ndmin=2)[:, :2]
    track = np.loadtxt(folder / f"{name}.track", dtype=float, ndmin=2)[:, :2]
    route = np.loadtxt(folder / f"{name}.route", dtype=int, ndmin=1)
    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError(f"Invalid node ID in {folder}")
    if np.any(route < 0) or np.any(route >= len(arcs)):
        raise ValueError(f"Invalid route arc ID in {folder}")
    return nodes, arcs, track, route


def candidate_features(folder: Path, k: int, max_points: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, arcs, track, route = read_example(folder)
    if max_points is not None and len(track) > max_points:
        # Spread samples across the whole trajectory; use every point for validation.
        track = track[np.linspace(0, len(track) - 1, max_points, dtype=int)]

    # Approximate metres for local map geometry. Avoid absolute coordinates so
    # features can transfer between the different maps in the dataset.
    latitude = np.deg2rad(np.mean(nodes[:, 1]))
    scale = np.array([111_320 * np.cos(latitude), 111_320.0])
    start = nodes[arcs[:, 0]] * scale
    end = nodes[arcs[:, 1]] * scale
    points = track * scale
    directions = end - start
    lengths = np.linalg.norm(directions, axis=1)
    midpoints = (start + end) / 2

    search = NearestNeighbors(n_neighbors=min(k, len(arcs)))
    search.fit(midpoints)
    _, ids = search.kneighbors(points)
    selected_start = start[ids]
    selected_direction = directions[ids]
    selected_length = lengths[ids]
    relative = points[:, None, :] - selected_start
    denominator = selected_length ** 2
    fraction = np.divide(
        np.sum(relative * selected_direction, axis=2),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = selected_start + fraction[:, :, None] * selected_direction
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2)

    movement = np.empty_like(points)
    if len(points) > 1:
        movement[0] = points[1] - points[0]
        movement[1:] = points[1:] - points[:-1]
    else:
        movement[:] = 0
    movement_length = np.linalg.norm(movement, axis=1)
    alignment = np.divide(
        np.sum(movement[:, None, :] * selected_direction, axis=2),
        movement_length[:, None] * selected_length,
        out=np.zeros_like(distance),
        where=(movement_length[:, None] * selected_length) > 0,
    )
    nearest_order = np.argsort(np.argsort(distance, axis=1), axis=1)
    features = np.stack(
        (np.log1p(distance), np.log1p(selected_length), alignment, fraction,
         nearest_order / max(ids.shape[1] - 1, 1)),
        axis=2,
    )
    labels = np.isin(ids, route).astype(int)
    return features, ids, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_dir", type=Path, help="Path to Student folder/train")
    parser.add_argument("--k", type=int, default=20, help="Candidate arcs per GPS point")
    parser.add_argument("--max-points", type=int, default=250,
                        help="GPS points sampled per training example")
    parser.add_argument("--max-iter", type=int, default=40,
                        help="Maximum MLP training iterations")
    args = parser.parse_args()
    if args.k < 1 or args.max_points < 1 or args.max_iter < 1:
        parser.error("--k, --max-points and --max-iter must be positive")
    examples = sorted(p for p in args.train_dir.iterdir() if p.is_dir())
    if len(examples) < 2:
        parser.error("Need at least two training examples")
    split = int(len(examples) * 0.8)
    training, validation = examples[:split], examples[split:]
    print(f"Training on {len(training)} examples; validating on {len(validation)}", flush=True)

    random = np.random.default_rng(42)
    feature_batches = []
    label_batches = []
    for number, example in enumerate(training, 1):
        features, _, labels = candidate_features(example, args.k, args.max_points)
        x = features.reshape(-1, features.shape[-1])
        y = labels.reshape(-1)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        keep_negatives = random.choice(negatives, size=min(len(negatives), 4 * max(len(positives), 1)), replace=False)
        keep = np.concatenate((positives, keep_negatives))
        feature_batches.append(x[keep])
        label_batches.append(y[keep])
        print(f"Loaded {number}/{len(training)}: {example.name}", flush=True)

    x_train = np.concatenate(feature_batches)
    y_train = np.concatenate(label_batches)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training candidates contain only one label class")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=args.max_iter,
                          early_stopping=True, random_state=42)
    model.fit(x_train, y_train)
    print(f"MLP trained on {len(y_train):,} candidate arcs", flush=True)

    scores = {}
    for number, example in enumerate(validation, 1):
        features, ids, labels = candidate_features(example, args.k)
        probabilities = model.predict_proba(scaler.transform(features.reshape(-1, features.shape[-1])))[:, 1]
        best = probabilities.reshape(ids.shape).argmax(axis=1)
        selected_labels = labels[np.arange(len(ids)), best]
        scores[example.name] = float(selected_labels.mean())
        print(f"Validated {number}/{len(validation)} {example.name}: {scores[example.name]:.1%}", flush=True)

    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "k": args.k}, output / "mlp_model.joblib")
    result = {
        "method": "MLP candidate-arc point matching",
        "score_definition": "fraction of GPS point arc matches present in reference route",
        "training_count": len(training), "validation_count": len(validation),
        "k": args.k, "max_points_per_training_example": args.max_points,
        "training_rows": int(len(y_train)), "validation_scores": scores,
        "mean_validation_score": float(np.mean(list(scores.values()))),
        "note": "Point matches are not a connected route; this is not the hackathon route metric.",
    }
    Path("mlp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Mean held-out validation score: {result['mean_validation_score']:.1%}")
    print("Saved mlp_results.json and artifacts/mlp_model.joblib")


if __name__ == "__main__":
    main()
