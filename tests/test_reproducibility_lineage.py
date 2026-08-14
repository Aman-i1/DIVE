"""Unit & Integration tests for Artifact Store, Lineage DAG & Reproducibility Bundle Exporter."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from sklearn.dummy import DummyClassifier

from dive.artifact_store import ArtifactStore, StoredArtifact
from dive.lineage import LineageGraph, LineageNode
from dive.reproducibility import ReproducibilityBundleExporter, ReproducibilityBundleMetadata


def test_artifact_store_content_addressing(tmp_path: Path) -> None:
    store = ArtifactStore(base_dir=tmp_path / "store")

    # Store bytes
    data = b"hello dive industrial automl"
    artifact = store.put_bytes(data, artifact_type="text")

    assert isinstance(artifact, StoredArtifact)
    assert len(artifact.artifact_hash) == 64  # SHA-256
    assert artifact.byte_size == len(data)

    # Retrieve bytes and verify integrity
    retrieved = store.get_bytes(artifact.artifact_hash)
    assert retrieved == data

    # Store pickle model
    model = DummyClassifier(strategy="most_frequent")
    model_art = store.put_pickle(model, artifact_type="model")
    loaded_model = store.get_pickle(model_art.artifact_hash)
    assert loaded_model is not None
    assert isinstance(loaded_model, DummyClassifier)


def test_lineage_graph_and_mermaid() -> None:
    dag = LineageGraph(experiment_id="exp_test_123")

    n1 = dag.add_node("node_data", "dataset", "telecom_churn.csv", artifact_hash="hash123")
    n2 = dag.add_node("node_feat", "feature_engineer", "LeakageSafeTemporalEngine", inputs=["node_data"])
    n3 = dag.add_node("node_model", "model_trial", "RandomForest", inputs=["node_feat"])
    n4 = dag.add_node("node_eval", "metrics", "F1: 0.92, ECE: 0.02", inputs=["node_model"])

    assert len(dag.nodes) == 4
    assert "node_data" in dag.nodes["node_feat"].inputs

    mermaid_code = dag.render_mermaid()
    assert "flowchart TD" in mermaid_code
    assert "node_data --> node_feat" in mermaid_code
    assert "node_feat --> node_model" in mermaid_code

    summary = dag.render_summary()
    assert "EXPERIMENT LINEAGE PROVENANCE" in summary


def test_reproducibility_bundle_export(tmp_path: Path) -> None:
    exporter = ReproducibilityBundleExporter(experiment_id="exp_bundle_42")

    model = DummyClassifier(strategy="most_frequent")
    dag = LineageGraph(experiment_id="exp_bundle_42")
    dag.add_node("d1", "dataset", "data.csv")

    bundle_dir = exporter.export_bundle(
        output_dir=tmp_path,
        model=model,
        target="churn",
        problem_type="classification",
        lineage=dag,
        random_seed=42,
    )

    assert bundle_dir.exists()
    assert (bundle_dir / "model.pkl").exists()
    assert (bundle_dir / "metadata.json").exists()
    assert (bundle_dir / "lineage.json").exists()
    assert (bundle_dir / "reproduce.py").exists()

    with open(bundle_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    assert meta["target"] == "churn"
    assert meta["random_seed"] == 42
