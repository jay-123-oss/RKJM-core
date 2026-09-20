"""Python HIL wrapper for the Verilator Watch Grid model.

The preferred path loads a shared library exposing ``watch_grid_run`` through
ctypes.  A Verilator executable is also supported, which keeps the workflow
usable when the simulator was built with ``--exe`` instead of a shared C ABI.
"""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from watch_grid_gold_model import WatchGrid2DSimulator


class WatchGridHIL:
    """Feed 4x4 tensors into either a ctypes library or Verilator binary."""

    def __init__(self, shared_library: Optional[Path] = None, executable: Optional[Path] = None) -> None:
        self.shared_library = shared_library
        self.executable = executable
        self._library = None
        if shared_library is not None:
            self._library = ctypes.CDLL(str(shared_library))
            self._library.watch_grid_run.argtypes = [
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.POINTER(ctypes.c_int32),
            ]
            self._library.watch_grid_run.restype = ctypes.c_int

    def run(self, activation: torch.Tensor, weight_code: torch.Tensor) -> torch.Tensor:
        """Execute one 4-feature vector and return four integer outputs."""

        activation_np = np.asarray(activation.detach().cpu(), dtype=np.int8).reshape(4)
        code_np = np.asarray(weight_code.detach().cpu(), dtype=np.uint8).reshape(4, 4)
        if not np.all(np.isin(activation_np, (-1, 1))):
            raise ValueError("RTL forward activation must contain -1/+1")
        if not np.all(np.isin(code_np, (0, 1, 3))):
            raise ValueError("weight codes must contain 0, 1, or 3")

        if self._library is not None:
            result = np.zeros(4, dtype=np.int32)
            status = self._library.watch_grid_run(
                activation_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                code_np.reshape(-1).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                result.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            if status != 0:
                raise RuntimeError(f"watch_grid_run returned status {status}")
            return torch.from_numpy(result.copy())

        if self.executable is None:
            raise RuntimeError("provide shared_library or executable")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as vector_file:
            vector_file.write("WATCH_GRID_COSIM_V1 4 4\n")
            vector_file.write("activations " + " ".join(map(str, activation_np.tolist())) + "\n")
            vector_file.write("weight_codes " + " ".join(map(str, code_np.reshape(-1).tolist())) + "\n")
            weights = np.where(code_np == 0, 0, np.where(code_np == 1, 1, -1)).astype(np.int32)
            expected = weights @ activation_np.astype(np.int32)
            vector_file.write("expected " + " ".join(map(str, expected.tolist())) + "\n")
            vector_file.flush()
            completed = subprocess.run(
                [str(self.executable), vector_file.name], capture_output=True, text=True, check=False
            )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        values = [int(line.split()[2]) for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
        if len(values) != 4:
            raise RuntimeError("Verilator executable did not emit four RESULT lines")
        return torch.tensor(values, dtype=torch.int32)


def validate_against_gold(hil: WatchGridHIL, seed: int = 23) -> None:
    rng = np.random.default_rng(seed)
    weights = rng.choice(np.asarray([-1, 0, 1], dtype=np.int8), size=(4, 4))
    activation = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=4)
    code = np.where(weights == 0, 0, np.where(weights == 1, 1, 3)).astype(np.uint8)
    expected = WatchGrid2DSimulator(weights.astype(np.float32)).forward(activation)
    actual = hil.run(torch.from_numpy(activation), torch.from_numpy(code))
    if not torch.equal(actual, torch.from_numpy(expected.astype(np.int32))):
        raise AssertionError(f"HIL mismatch: expected={expected.tolist()} actual={actual.tolist()}")
    print(f"HIL PASS bit-exact=4/4 outputs={actual.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-library", type=Path)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    validate_against_gold(WatchGridHIL(args.shared_library, args.executable))