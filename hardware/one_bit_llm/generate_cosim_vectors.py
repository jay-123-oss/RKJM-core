"""Generate deterministic RTL co-simulation vectors from the Python Gold Model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from watch_grid_gold_model import WatchGrid2DSimulator


def generate(path: Path, seed: int = 19) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.choice(np.asarray([-1, 0, 1], dtype=np.int8), size=(4, 4))
    activations = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=4)
    expected = WatchGrid2DSimulator(weights.astype(np.float32)).forward(activations)

    with path.open("w", encoding="ascii") as vector_file:
        vector_file.write("WATCH_GRID_COSIM_V1 4 4\n")
        vector_file.write("activations " + " ".join(str(int(value)) for value in activations) + "\n")
        vector_file.write("weight_codes ")
        codes = {0: 0, 1: 1, -1: 3}
        vector_file.write(
            " ".join(str(codes[int(value)]) for value in weights.reshape(-1)) + "\n"
        )
        vector_file.write("expected " + " ".join(str(int(value)) for value in expected) + "\n")

    print(f"wrote {path}")
    print(f"activations={activations.tolist()}")
    print(f"weights={weights.tolist()}")
    print(f"expected={expected.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("watch_grid_vectors.txt"))
    parser.add_argument("--seed", type=int, default=19)
    arguments = parser.parse_args()
    generate(arguments.path, arguments.seed)