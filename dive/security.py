"""Security, Path Traversal & Safe Deserialization Auditor - `dive/security.py`.

Enforces security best practices:
1. Path Traversal Guard: Sanitizes relative paths to prevent directory traversal attacks (../../).
2. Safe Deserialization Gate: Inspects serialized artifacts and model files for unsafe bytecode instructions.
3. Cryptographic Checksum Verification: Validates SHA-256 integrity before loading remote/local artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class SecurityAuditResult:
    """Outcome of security inspection."""

    is_secure: bool
    risk_level: str  # 'SAFE', 'WARNING', 'CRITICAL_RISK'
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_secure": self.is_secure,
            "risk_level": self.risk_level,
            "warnings": self.warnings,
        }


class SecurityAuditor:
    """Sanitizes paths, validates integrity, and blocks unsafe deserialization payloads."""

    # Unsafe global modules to block during unpickling inspection
    BLOCKED_MODULES = frozenset({"os", "subprocess", "posix", "nt", "shutil", "builtins.exec", "builtins.eval"})

    @staticmethod
    def safe_path_join(base_directory: Union[str, Path], user_path: Union[str, Path]) -> Path:
        """Resolve path strictly inside base_directory, raising ValueError on path traversal."""
        base = Path(base_directory).resolve()
        target = (base / user_path).resolve()

        # Check if target is inside base
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(
                f"Security Violation: Path traversal detected. '{user_path}' attempts to escape base directory '{base}'."
            )
        return target

    @classmethod
    def audit_pickle_bytes(cls, data: bytes) -> SecurityAuditResult:
        """Inspect raw pickle bytecode for malicious system execution calls."""
        warnings: List[str] = []

        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                full_name = f"{module}.{name}"
                if module in ("os", "subprocess", "posix", "nt", "shutil") or full_name in cls.BLOCKED_MODULES:
                    warnings.append(f"Unsafe module execution detected: '{full_name}'")
                    raise pickle.UnpicklingError(f"Blocked unsafe module: {full_name}")
                return super().find_class(module, name)

        try:
            SafeUnpickler(io.BytesIO(data)).load()
        except pickle.UnpicklingError:
            return SecurityAuditResult(is_secure=False, risk_level="CRITICAL_RISK", warnings=warnings)
        except Exception:
            # Deserialization error or missing classes is acceptable if no security violations were raised
            pass

        if warnings:
            return SecurityAuditResult(is_secure=False, risk_level="CRITICAL_RISK", warnings=warnings)

        return SecurityAuditResult(is_secure=True, risk_level="SAFE", warnings=[])

    @staticmethod
    def verify_file_sha256(file_path: Union[str, Path], expected_sha256: str) -> bool:
        """Verify that the cryptographic SHA-256 hash matches the expected signature."""
        path = Path(file_path)
        if not path.exists():
            return False

        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)

        return hasher.hexdigest().lower() == expected_sha256.lower()
