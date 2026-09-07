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
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Set to 1/true/yes to forbid every on-demand install (CI, locked-down hosts).
NO_INSTALL_ENV = "DIVE_NO_INSTALL"


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
    #: Approximate wheel download size in MB, quoted before an install prompt.
    #: A user agreeing to "install torch" deserves to know it is not 2 MB.
    download_mb: Optional[int] = None
    #: On-demand packages are excluded from the startup "missing extras" notice.
    #: A tabular-ML user should not be nagged about video decoders they will
    #: never use; ``ensure_available`` asks for them at the point of need.
    on_demand: bool = False
    #: Last ``purpose`` string a caller passed to :func:`load_optional`, so a
    #: missing-dependency message can name the feature that wanted it.
    last_purpose: Optional[str] = None

    @property
    def install_hint(self) -> str:
        return f"pip install {self.name}"

    @property
    def size_hint(self) -> str:
        return f"~{self.download_mb} MB download" if self.download_mb else "size unknown"


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
    "sentence_transformers": OptionalPackage(
        name="sentence_transformers",
        import_name="sentence_transformers",
        extra="nlp",
        provides="dense neural text embeddings (SentenceTransformers)",
    ),
    "transformers": OptionalPackage(
        name="transformers",
        import_name="transformers",
        extra="nlp",
        provides="Hugging Face Transformer architectures and tokenizers",
    ),
    "torch": OptionalPackage(
        name="torch",
        import_name="torch",
        extra="dl",
        provides="PyTorch deep learning tensor runtime",
        download_mb=250,
        on_demand=True,
    ),
    "torchvision": OptionalPackage(
        name="torchvision",
        import_name="torchvision",
        extra="cv",
        provides="pretrained image backbones and image transforms",
        download_mb=30,
        on_demand=True,
    ),
    "torchaudio": OptionalPackage(
        name="torchaudio",
        import_name="torchaudio",
        extra="dl",
        provides="audio loading and spectrogram transforms on tensors",
        download_mb=10,
        on_demand=True,
    ),
    "timm": OptionalPackage(
        name="timm",
        import_name="timm",
        extra="cv",
        provides="large catalogue of pretrained vision backbones",
        download_mb=5,
        on_demand=True,
    ),
    "librosa": OptionalPackage(
        name="librosa",
        import_name="librosa",
        extra="dl",
        provides="audio decoding, resampling and log-mel features",
        download_mb=25,
        on_demand=True,
    ),
    "soundfile": OptionalPackage(
        name="soundfile",
        import_name="soundfile",
        extra="dl",
        provides="WAV/FLAC/OGG reading without an external ffmpeg",
        download_mb=2,
        on_demand=True,
    ),
    "av": OptionalPackage(
        name="av",
        import_name="av",
        extra="dl",
        provides="video container demuxing and frame decoding (PyAV)",
        download_mb=35,
        on_demand=True,
    ),
    "opencv-python": OptionalPackage(
        name="opencv-python",
        import_name="cv2",
        extra="cv",
        provides="image/video decoding and resizing (OpenCV)",
        download_mb=65,
        on_demand=True,
    ),
    "pillow": OptionalPackage(
        name="pillow",
        import_name="PIL",
        extra="cv",
        provides="image decoding and resizing for the no-torch fallback",
        download_mb=3,
    ),
    "onnxruntime": OptionalPackage(
        name="onnxruntime",
        import_name="onnxruntime",
        extra="serving",
        provides="high-performance ONNX model inference runtime",
    ),
    "skl2onnx": OptionalPackage(
        name="skl2onnx",
        import_name="skl2onnx",
        extra="serving",
        provides="Scikit-Learn to ONNX model graph converter",
    ),
    "psutil": OptionalPackage(
        name="psutil",
        import_name="psutil",
        extra="ops",
        provides="live RAM/CPU detection for resource-aware planning "
        "(falls back to conservative defaults)",
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


def load_optional(name: str, purpose: Optional[str] = None) -> Optional[Any]:
    """Return the imported module, or ``None`` when it is unavailable.

    ``purpose`` describes what the caller wanted the module for. It is accepted
    because four call sites already pass it - ``dive/nlp/embeddings/representation.py``,
    ``dive/nlp/transformers/estimator.py`` (twice) and ``dive/nlp/optimization/onnx.py``
    - and without the parameter every one of them raised ``TypeError`` inside a
    ``try/except Exception`` that degraded to a fallback. The result was that dense
    embeddings, transformer fine-tuning and ONNX inference were *silently dead even
    when their package was installed*. The argument is recorded on the package so
    ``dive deps`` can say what a missing extra was actually wanted for.
    """
    _probe()
    pkg = _REGISTRY.get(name)
    if pkg is None:
        return None
    if purpose:
        pkg.last_purpose = str(purpose)
    return pkg.module if pkg.available else None


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

    On-demand packages are omitted: they are large, domain-specific downloads
    that :func:`ensure_available` offers at the moment they are needed. Listing
    them on every ``dive`` invocation would nag a tabular-ML user about video
    decoders. ``dive deps`` still reports the full registry.
    """
    missing = [pkg for pkg in missing_packages() if not pkg.on_demand]
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


# ----------------------------------------------------------------------
# on-demand installation
# ----------------------------------------------------------------------
def installs_forbidden() -> bool:
    """True when the environment forbids installing anything."""
    return os.environ.get(NO_INSTALL_ENV, "").strip().lower() in {"1", "true", "yes"}


def refresh(name: str) -> bool:
    """Re-probe a single package, e.g. after installing it.

    ``_probe`` runs once per process, so a package installed mid-run would
    otherwise still be reported as missing.
    """
    pkg = _REGISTRY.get(name)
    if pkg is None:
        return False
    # Drop cached negative import results; a fresh install adds a new path entry.
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(pkg.import_name)
    except Exception as exc:
        pkg.available = False
        pkg.module = None
        pkg.error = f"{type(exc).__name__}: {exc}"
        return False
    pkg.available = True
    pkg.module = module
    pkg.version = getattr(module, "__version__", None)
    pkg.error = None
    _configure(pkg)
    return True


def _confirm(question: str, console: Any = None) -> bool:
    """Ask a yes/no question, defaulting to *no* on anything unattended.

    A non-interactive stdin (CI, a piped script, a service) must never block on
    a prompt, and must never install silently either - so it declines.
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return False
    except Exception:
        return False
    try:
        import click

        return bool(click.confirm(question, default=False))
    except ImportError:
        pass
    try:
        answer = input(f"{question} [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        return False
    return answer.strip().lower() in {"y", "yes"}


def ensure_available(
    name: str,
    console: Any = None,
    assume_yes: bool = False,
    allow_install: bool = True,
) -> bool:
    """Make ``name`` importable, installing it on demand with consent.

    The dependency policy this implements, in order:

    1. **Never reinstall.** An importable package returns ``True`` immediately;
       no pip process is started and nothing is printed.
    2. **The user decides.** Absent packages are named along with what they
       unlock and roughly how large the download is, then confirmed. Declining
       is a normal outcome that returns ``False`` - callers fall back rather
       than fail.
    3. **Never install unattended.** A non-interactive stdin, ``assume_yes``
       unset, or ``DIVE_NO_INSTALL=1`` all mean "do not install"; the manual
       command is printed instead.

    Returns ``True`` only when the package is importable at the end.
    """
    _probe()
    pkg = _REGISTRY.get(name)
    if pkg is None:
        if console is not None:
            console.warn(f"'{name}' is not a registered optional dependency.")
        return False
    if pkg.available:
        return True

    if not allow_install or installs_forbidden():
        if console is not None:
            reason = (
                f"{NO_INSTALL_ENV} is set"
                if installs_forbidden()
                else "installation is disabled for this command"
            )
            console.warn(
                f"{pkg.name} is required for {pkg.provides} but {reason}. "
                f"Install it yourself with: {pkg.install_hint}"
            )
        return False

    question = (
        f"Install {pkg.name} ({pkg.size_hint}) to enable {pkg.provides}?"
    )
    if not assume_yes and not _confirm(question, console):
        if console is not None:
            console.warn(
                f"Skipping {pkg.name}. Install it later with: {pkg.install_hint}"
            )
        return False

    if console is not None:
        console.info(f"  Installing {pkg.name} ({pkg.size_hint}) - this may take a while...")
    command = [sys.executable, "-m", "pip", "install", pkg.name]
    try:
        completed = subprocess.run(command, check=False)
    except Exception as exc:
        if console is not None:
            console.error(f"Could not run pip: {exc}. Install manually: {pkg.install_hint}")
        return False

    if completed.returncode != 0:
        if console is not None:
            console.error(
                f"pip failed to install {pkg.name} (exit {completed.returncode}). "
                f"Install manually: {pkg.install_hint}"
            )
        return False

    if not refresh(name):
        if console is not None:
            console.error(
                f"{pkg.name} installed but still not importable ({pkg.error}). "
                "A restart of the interpreter may be required."
            )
        return False

    if console is not None:
        console.success(f"{pkg.name} {pkg.version or ''} is ready.".replace("  ", " "))
    return True


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
