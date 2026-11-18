"""First map-matching baseline for one example in the supplied dataset.

KNN indexes road-segment midpoints. For each GPS point, we inspect its k
nearest segments and choose the segment with the smallest point-to-segment
distance. This does not yet connect the selected segments into a full route.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def read_table(path: Path, dtype: type) -> np.ndarray:
    data = np.loadtxt(path, dtype=dtype, ndmin=2)
    if data.size == 0:
        raise ValueError(f"Empty file: {path}")
    return data


def match_example(example: Path, k: int) -> np.ndarray:
    name = example.name
    nodes = read_table(example / f"{name}.nodes", float)[:, :2]
    arcs = read_table(example / f"{name}.arcs", int)[:, :2]
    track = read_table(example / f"{name}.track", float)[:, :2]

    if np.any(arcs < 0) or np.any(arcs >= len(nodes)):
        raise ValueError("An arc refers to a node outside the .nodes file")

    # Scale longitude so one unit has roughly the same distance as latitude.
    scale = np.array([np.cos(np.deg2rad(np.mean(nodes[:, 1]))), 1.0])
    start = nodes[arcs[:, 0]] * scale
    end = nodes[arcs[:, 1]] * scale
    points = track * scale
    midpoints = (start + end) / 2

    search = NearestNeighbors(n_neighbors=min(k, len(arcs)), algorithm="auto")
    search.fit(midpoints)
    _, candidate_ids = search.kneighbors(points)

    candidate_start = start[candidate_ids]
    candidate_end = end[candidate_ids]
    direction = candidate_end - candidate_start
    length_squared = np.sum(direction * direction, axis=2)
    relative = points[:, None, :] - candidate_start
    projection = np.divide(
        np.sum(relative * direction, axis=2),
        length_squared,
        out=np.zeros_like(length_squared),
        where=length_squared > 0,
    )
    projection = np.clip(projection, 0, 1)
    closest = candidate_start + projection[:, :, None] * direction
    distance_squared = np.sum((points[:, None, :] - closest) ** 2, axis=2)
    best = np.argmin(distance_squared, axis=1)
    return candidate_ids[np.arange(len(points)), best]


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path, help="Path to one numbered example folder")
    parser.add_argument("--k", type=int, default=10, help="Candidate segments per GPS point")
    parser.add_argument("--output", type=Path, help="Optional output text file")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    predicted = match_example(args.example, args.k)
    print(f"Matched {len(predicted)} GPS points to road arcs")
    print("First 20 predicted arc IDs:", predicted[:20].tolist())

    route_file = args.example / f"{args.example.name}.route"
    if route_file.exists():
        reference = np.loadtxt(route_file, dtype=int, ndmin=1)
        hit_rate = np.isin(predicted, reference).mean()
        print(f"GPS point matches on reference route: {hit_rate:.1%}")
        print("This is a rough baseline score, not the hackathon's route metric.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.output, predicted, fmt="%d")
        print(f"Saved point matches to {args.output}")


if __name__ == "__main__":
    main()
