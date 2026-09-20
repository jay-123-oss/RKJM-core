"""Build and run the Verilator bit-exact Phase 2 co-simulation."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from generate_cosim_vectors import generate


def run(workspace: Path, rebuild: bool = True) -> None:
    verilator = shutil.which("verilator")
    if verilator is None:
        raise RuntimeError(
            "Verilator is not installed. Install Verilator, then rerun this command; "
            "Python vector generation remains available independently."
        )

    vectors = workspace / "watch_grid_vectors.txt"
    generate(vectors)
    obj_dir = workspace / "obj_dir"
    executable = obj_dir / "Vwatch_grid_4x4"
    if rebuild or not executable.exists():
        command = [
            verilator,
            "--cc",
            "--exe",
            "--build",
            "--top-module",
            "watch_grid_4x4",
            "-Mdir",
            str(obj_dir),
            str(workspace / "watch_grid_4x4.sv"),
            str(workspace / "tb_watch_grid_co_sim.cpp"),
        ]
        subprocess.run(command, cwd=workspace, check=True)
    subprocess.run([str(executable), str(vectors)], cwd=workspace, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()
    run(Path(__file__).resolve().parent, rebuild=not args.no_rebuild)