from __future__ import annotations

import hashlib, json, os, platform, random, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    # Required by deterministic CUDA matrix multiplications on CUDA >= 10.2.
    # Setting it before the first CUDA operation keeps GPU runs reproducible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def manifest(seed: int, config_path: Path, command: str) -> dict:
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "seed_python": seed,
            "seed_numpy": seed, "seed_torch": seed, "seed_synthea": seed,
            "python": sys.version, "platform": platform.platform(), "command": command,
            "config_hash": file_hash(config_path), "project_commit": "unavailable-empty-git-metadata",
            "paper": "arXiv:2602.03517v2", "hardware": platform.processor(),
            "torch": torch.__version__, "cwd": os.getcwd()}


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
