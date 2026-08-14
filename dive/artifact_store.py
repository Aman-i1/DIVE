"""Content-Addressable Cryptographic Artifact Store - `dive/artifact_store.py`.

Stores models, datasets, preprocessors, and metadata keyed by their SHA-256 hash.
Provides automatic deduplication and cryptographic integrity verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class StoredArtifact:
    """Descriptor for a stored content-addressable artifact."""

    artifact_hash: str
    artifact_type: str  # 'model', 'dataset', 'preprocessor', 'metrics', 'report'
    byte_size: int
    storage_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "artifact_type": self.artifact_type,
            "byte_size": self.byte_size,
            "storage_path": str(self.storage_path),
            "metadata": self.metadata,
        }


class ArtifactStore:
    """Content-addressable storage engine indexed by cryptographic SHA-256 hashes."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path("./.dive_store")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.base_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)

    def put_bytes(
        self,
        data: bytes,
        artifact_type: str = "binary",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Store raw bytes addressable by SHA-256."""
        sha = hashlib.sha256(data).hexdigest()
        dest = self.base_dir / f"{sha}.bin"

        if not dest.exists():
            with open(dest, "wb") as f:
                f.write(data)

        meta = metadata or {}
        self._index[sha] = {
            "artifact_hash": sha,
            "artifact_type": artifact_type,
            "byte_size": len(data),
            "storage_path": str(dest),
            "metadata": meta,
        }
        self._save_index()

        return StoredArtifact(
            artifact_hash=sha,
            artifact_type=artifact_type,
            byte_size=len(data),
            storage_path=dest,
            metadata=meta,
        )

    def put_pickle(
        self,
        obj: Any,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredArtifact:
        """Serialize object with pickle and store content-addressably."""
        data = pickle.dumps(obj)
        return self.put_bytes(data, artifact_type=artifact_type, metadata=metadata)

    def get_bytes(self, artifact_hash: str) -> Optional[bytes]:
        """Retrieve raw bytes by SHA-256 hash."""
        dest = self.base_dir / f"{artifact_hash}.bin"
        if dest.exists():
            with open(dest, "rb") as f:
                data = f.read()
            # Verify cryptographic integrity
            if hashlib.sha256(data).hexdigest() == artifact_hash:
                return data
        return None

    def get_pickle(self, artifact_hash: str) -> Optional[Any]:
        """Retrieve and deserialize pickled object."""
        data = self.get_bytes(artifact_hash)
        if data is not None:
            try:
                return pickle.loads(data)
            except Exception:
                return None
        return None

    def list_artifacts(self) -> List[StoredArtifact]:
        """List all stored artifacts in the store."""
        return [
            StoredArtifact(
                artifact_hash=item["artifact_hash"],
                artifact_type=item["artifact_type"],
                byte_size=item["byte_size"],
                storage_path=Path(item["storage_path"]),
                metadata=item.get("metadata", {}),
            )
            for item in self._index.values()
        ]
