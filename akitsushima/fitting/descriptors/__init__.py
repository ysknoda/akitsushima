"""In-package descriptors for the MLIP models (ADR-0004, decision 4)."""

from .acsf import ACSF, ATOMIC_NUMBER, build_table, sort_species, write_table_csv

__all__ = [
    "ACSF",
    "ATOMIC_NUMBER",
    "build_table",
    "descriptor_from_config",
    "sort_species",
    "write_table_csv",
]


def descriptor_from_config(cfg):
    """Rebuild a descriptor from the config stored with a model."""
    import json

    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    kind = cfg.get("type", "acsf")
    if kind == "acsf":
        return ACSF.from_config(cfg)
    raise ValueError(f"descriptor_from_config: unknown type {kind!r}.")
