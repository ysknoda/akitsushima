"""Optional PyYAML detection, isolated in its own module."""
try:
    import yaml as _yaml          # PyYAML (pip install pyyaml)
    _HAS_YAML = True
except ImportError:
    _yaml = None
    _HAS_YAML = False

__all__ = ["_HAS_YAML", "_yaml"]
