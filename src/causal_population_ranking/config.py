from __future__ import annotations

from pathlib import Path
import yaml


def load_prometheus_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = {"run", "data", "simulation", "nuisance", "prometheus"}
    missing = required.difference(cfg or {})
    if missing:
        raise ValueError(f"Missing PROMETHEUS config sections: {sorted(missing)}")
    if int(cfg["run"]["seed"]) < 0 or int(cfg["nuisance"]["folds"]) < 2:
        raise ValueError("Invalid seed or fold count")
    # The integrated DM77 action path is the application default. Preserved
    # numeric-transition experiments declare synthetic_legacy explicitly.
    cfg.setdefault("opportunity_source", "dm77_catalog")
    return cfg
