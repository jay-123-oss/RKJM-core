"""
Bootstrap & Dependency Verification Engine for Qwen Architecture in RKMJ-Core.
Ensures torch, safetensors, transformers, huggingface_hub, and psutil are all present.
"""

import importlib
import logging
import subprocess
import sys
from typing import Dict, List, Tuple

logger = logging.getLogger("qween.bootstrap")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] [qween.bootstrap] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

REQUIRED_DEPENDENCIES = [
    ("torch", "torch>=2.0.0"),
    ("safetensors", "safetensors>=0.4.0"),
    ("transformers", "transformers>=4.40.0"),
    ("huggingface_hub", "huggingface_hub>=0.20.0"),
    ("psutil", "psutil>=5.8.0"),
]


def check_dependencies() -> Tuple[List[str], Dict[str, str]]:
    """
    Checks if required dependencies are present and returns their versions.
    Returns (missing_packages, installed_versions).
    """
    missing = []
    installed = {}
    for mod_name, pip_spec in REQUIRED_DEPENDENCIES:
        try:
            mod = importlib.import_module(mod_name)
            installed[mod_name] = getattr(mod, "__version__", "installed")
        except ImportError:
            missing.append(pip_spec)
    return missing, installed


def ensure_dependencies(auto_install: bool = True) -> bool:
    """
    Verifies that all required dependencies are installed.
    If any are missing and auto_install=True, attempts to install them via pip.
    """
    missing, installed = check_dependencies()
    if not missing:
        return True

    logger.warning("Missing required dependencies: %s", missing)

    if auto_install:
        logger.info("Attempting automatic 1-click bootstrap installation...")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing
            logger.info("Executing: %s", " ".join(cmd))
            subprocess.check_call(cmd)
            # Re-verify
            re_missing, _ = check_dependencies()
            if not re_missing:
                logger.info("✅ All dependencies successfully installed!")
                return True
        except Exception as e:
            logger.error("Auto-install failed: %s", e)

    msg = (
        "\n" + "=" * 60 + "\n"
        "❌ MISSING DEPENDENCIES DETECTED!\n"
        f"Missing: {', '.join(missing)}\n\n"
        "Please activate the virtual environment and install them:\n"
        "    source myenv/bin/activate\n"
        f"    pip install {' '.join(missing)}\n" + "=" * 60 + "\n"
    )
    raise RuntimeError(msg)


def main():
    print("=" * 60)
    print("🔍 RKMJ-Core Qwen Engine: Dependency Verification")
    print(f"   Python Executable: {sys.executable}")
    print(f"   Python Version:    {sys.version.split()[0]}")
    print("=" * 60)

    missing, installed = check_dependencies()
    for mod_name, _ in REQUIRED_DEPENDENCIES:
        if mod_name in installed:
            print(f"  [✓] {mod_name:<16} : {installed[mod_name]}")
        else:
            print(f"  [✗] {mod_name:<16} : NOT FOUND")

    if missing:
        print("\nResolving missing dependencies...")
        ensure_dependencies(auto_install=True)
    else:
        print("\n✅ All 5 required dependencies are present and verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
