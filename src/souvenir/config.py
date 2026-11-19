"""Load configs/pipeline.yaml (with ${ENV} interpolation and a .env file) and product specs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from souvenir.schemas import ProductSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader. Existing environment variables win."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        out = _ENV_PATTERN.sub(repl, value)
        return out if out != "" else None
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


class Config:
    """Thin wrapper around the nested pipeline.yaml dict."""

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        self.data = data
        self.root = root

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "Config":
        load_dotenv()
        path = Path(path) if path else PROJECT_ROOT / "configs" / "pipeline.yaml"
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = _interpolate(data)
        for dotted, value in (overrides or {}).items():
            cls._set(data, dotted, value)
        return cls(data)

    @staticmethod
    def _set(data: dict[str, Any], dotted: str, value: Any) -> None:
        node = data
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for p in dotted.split("."):
            if not isinstance(node, dict) or p not in node:
                return default
            node = node[p]
        return default if node is None else node

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.get(name, {}) or {})

    def path(self, value: str | None) -> Path | None:
        """Resolve a config path relative to the project root."""
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else (self.root / p)


def load_product(path: str | Path) -> ProductSpec:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    return ProductSpec.model_validate(json.loads(p.read_text(encoding="utf-8")))


def resolve_asset(spec_value: str) -> Path:
    p = Path(spec_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_prompt(name: str) -> str:
    return (PROJECT_ROOT / "prompts" / name).read_text(encoding="utf-8")
