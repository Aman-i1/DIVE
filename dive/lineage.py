"""Immutable Experiment Lineage & Provenance Graph - `dive/lineage.py`.

Tracks complete end-to-end directed acyclic graph (DAG) of the ML pipeline:
Raw Data Snapshot -> Preprocessing Pipeline -> Feature Selection -> Trial Search -> Evaluation Metrics -> Audit Certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import json
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LineageNode:
    """A single node in the provenance graph."""

    node_id: str
    node_type: str  # 'dataset', 'preprocessor', 'feature_engineer', 'model_trial', 'metrics', 'certificate'
    name: str
    artifact_hash: Optional[str] = None
    inputs: List[str] = field(default_factory=list)  # List of predecessor node_ids
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "artifact_hash": self.artifact_hash,
            "inputs": self.inputs,
            "attributes": self.attributes,
            "timestamp": self.timestamp,
        }


class LineageGraph:
    """Directed Acyclic Graph (DAG) representing the complete lineage of an experiment."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = experiment_id
        self.nodes: Dict[str, LineageNode] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        name: str,
        artifact_hash: Optional[str] = None,
        inputs: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Add a lineage node to the DAG."""
        node = LineageNode(
            node_id=node_id,
            node_type=node_type,
            name=name,
            artifact_hash=artifact_hash,
            inputs=inputs or [],
            attributes=attributes or {},
        )
        self.nodes[node_id] = node
        return node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "node_count": len(self.nodes),
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    def render_mermaid(self) -> str:
        """Render the lineage DAG into Mermaid flowchart syntax."""
        lines = ["flowchart TD", f"  subgraph Experiment_{self.experiment_id}"]
        for node in self.nodes.values():
            label = f"{node.name}\\n[{node.node_type}]"
            lines.append(f'    {node.node_id}["{label}"]')

        for node in self.nodes.values():
            for inp_id in node.inputs:
                if inp_id in self.nodes:
                    lines.append(f"    {inp_id} --> {node.node_id}")

        lines.append("  end")
        return "\n".join(lines)

    def render_summary(self) -> str:
        """Render human-readable text summary of lineage."""
        lines = [
            f"EXPERIMENT LINEAGE PROVENANCE: {self.experiment_id}",
            "==================================================",
        ]
        for node in self.nodes.values():
            inps = f" <- [{', '.join(node.inputs)}]" if node.inputs else ""
            lines.append(f"  [{node.node_type.upper():<16}] {node.name} (id: {node.node_id}){inps}")
            if node.artifact_hash:
                lines.append(f"    artifact SHA-256: {node.artifact_hash}")
        return "\n".join(lines)
