"""Central registry for optional third-party dependencies.

The notebook this package grew out of guarded every optional import with a
``try/except ImportError``. That pattern is preserved here, but centralised so
that:

* a single import site decides availability, and every module agrees;
* the CLI can tell the user *which* extras are missing, *what* they cost, and
  *how* to install them, instead of silently training a smaller model zoo.

Only ``numpy``/``pandas``/``scikit-learn``/``matplotlib``/``click``/``PyYAML``
are hard requirements. Everything below is optional.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OptionalPackage:
    """Metadata describing one optional dependency."""

    name: str
    import_name: str
    extra: str
    provides: str
    module: Any = field(default=None, repr=False)
    version: Optional[str] = None
    available: bool = False
    error: Optional[str] = None

    @property
    def install_hint(self) -> str:
        return f"pip install {self.name}"


# Registry order is the order shown in `dive deps` / startup warnings.
_REGISTRY: Dict[str, OptionalPackage] = {
    "xgboost": OptionalPackage(
        name="xgboost",
        import_name="xgboost",
        extra="boosters",
        provides="XGBoost gradient-boosting model",
    ),
    "lightgbm": OptionalPackage(
        name="lightgbm",
        import_name="lightgbm",
        extra="boosters",
        provides="LightGBM gradient-boosting model",
    ),
    "catboost": OptionalPackage(
        name="catboost",
        import_name="catboost",
        extra="boosters",
        provides="CatBoost gradient-boosting model",
    ),
    "optuna": OptionalPackage(
        name="optuna",
        import_name="optuna",
        extra="tuning",
        provides="hyperparameter tuning (balanced/competition modes)",
    ),
    "category_encoders": OptionalPackage(
        name="category_encoders",
        import_name="category_encoders",
        extra="explain",
        provides="target encoding of categorical features",
    ),
    "shap": OptionalPackage(
        name="shap",
        import_name="shap",
        extra="explain",
        provides="SHAP feature importance (falls back to native importances)",
    ),
}

_PROBED = False


def _probe() -> None:
    """Attempt to import every optional package exactly once per process."""
    global _PROBED
    if _PROBED:
        return
    for pkg in _REGISTRY.values():
        try:
            module = importlib.import_module(pkg.import_name)
        except Exception as exc:  # ImportError, but also broken installs
            pkg.available = False
            pkg.error = f"{type(exc).__name__}: {exc}"
            pkg.module = None
        else:
            pkg.available = True
            pkg.module = module
            pkg.version = getattr(module, "__version__", None)
            _configure(pkg)
    _PROBED = True


def _configure(pkg: OptionalPackage) -> None:
    """Apply per-package settings that must run right after a successful import."""
    if pkg.import_name == "optuna":
        try:
            pkg.module.logging.set_verbosity(pkg.module.logging.WARNING)
        except Exception:
            pass


def is_available(name: str) -> bool:
    """Return True if the named optional package imported successfully."""
    _probe()
    pkg = _REGISTRY.get(name)
    return bool(pkg and pkg.available)


def load_optional(name: str) -> Optional[Any]:
    """Return the imported module, or ``None`` when it is unavailable."""
    _probe()
    pkg = _REGISTRY.get(name)
    return pkg.module if pkg and pkg.available else None


def version_tuple(name: str, parts: int = 2) -> Tuple[int, ...]:
    """Return the package version as an int tuple, or zeros when unavailable.

    Used for version-gated behaviour such as the XGBoost >= 2.0 early-stopping
    API change. Non-numeric version segments (``2.0.0rc1``) degrade to 0 rather
    than raising.
    """
    _probe()
    pkg = _REGISTRY.get(name)
    if not pkg or not pkg.available or not pkg.version:
        return tuple([0] * parts)
    out: List[int] = []
    for segment in str(pkg.version).split(".")[:parts]:
        digits = ""
        for char in segment:
            if char.isdigit():
                digits += char
            else:
                break
        out.append(int(digits) if digits else 0)
    while len(out) < parts:
        out.append(0)
    return tuple(out)


def dependency_report() -> List[OptionalPackage]:
    """Return every registered optional package with its resolved status."""
    _probe()
    return list(_REGISTRY.values())


def missing_packages() -> List[OptionalPackage]:
    """Return only the optional packages that failed to import."""
    return [pkg for pkg in dependency_report() if not pkg.available]


def missing_summary() -> str:
    """Return a multi-line, human-readable summary of skipped capabilities.

    Returns an empty string when every optional package is installed, so callers
    can do ``if summary: console.warn(summary)``.
    """
    missing = missing_packages()
    if not missing:
        return ""
    lines = [
        f"{len(missing)} optional package(s) not installed - "
        "continuing with reduced capability:"
    ]
    for pkg in missing:
        lines.append(f"  - {pkg.name}: skipping {pkg.provides}  ({pkg.install_hint})")
    extras = sorted({pkg.extra for pkg in missing})
    lines.append(f"  Install all at once: pip install 'dive[{','.join(extras)}]'")
    return "\n".join(lines)


def detect_gpu() -> bool:
    """Best-effort NVIDIA GPU detection, safe on every platform.

    Uses ``shutil.which`` before invoking anything so that a missing
    ``nvidia-smi`` costs nothing and never spawns a shell. Returns False on any
    failure - GPU support is an optimisation, never a requirement.

    Set ``AUTOML_NO_GPU=1`` to force CPU-only training (used by CI).
    """
    import os
    import shutil
    import subprocess

    if os.environ.get("AUTOML_NO_GPU", "").strip().lower() in {"1", "true", "yes"}:
        return False

    executable = shutil.which("nvidia-smi")
    if not executable:
        return False
    try:
        completed = subprocess.run(
            [executable, "--list-gpus"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return completed.returncode == 0 and bool(completed.stdout.strip())
    except Exception:
        return False


# Public alias used across the package.
OPTIONAL_PACKAGES = _REGISTRY
