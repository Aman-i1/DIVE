"""CLI Command logic for `dive gate`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dive.commands.predict import _load_artifact
from dive.gate import DeploymentGate
from dive.predictor import DivePredictor
from dive.utils.io import load_dataframe, save_json
from dive.utils.logging import Console


def run_gate(
    console: Console,
    model_path: str,
    data_path: str,
    ref_path: Optional[str] = None,
    output_path: Optional[str] = None,
    strict: bool = False,
) -> int:
    """Run production deployment gate evaluation; returns exit code 0 (PASS) or 1 (FAIL)."""
    console.banner("🚪 DIVE PRODUCTION DEPLOYMENT GATE", f"Evaluating model: {model_path}")

    artifact = _load_artifact(model_path)
    if isinstance(artifact, DivePredictor):
        predictor = artifact
    else:
        best_est = artifact.best_estimator_
        feat_eng = artifact.feature_engineer_
        feat_cols = artifact.feature_columns_
        target = artifact.target
        prob_type = artifact.problem_type
        label_enc = getattr(artifact, "label_encoder_", None)
        schema = artifact._metadata.get("schema", {})
        predictor = DivePredictor(
            model_name=artifact.best_model_name_,
            estimator=best_est,
            feature_engineer=feat_eng,
            feature_columns=feat_cols,
            label_encoder=label_enc,
            target=target,
            problem_type=prob_type,
            input_schema=schema,
        )

    current_df = load_dataframe(data_path)
    ref_df = load_dataframe(ref_path) if ref_path else None

    gate = DeploymentGate(strict=strict)
    verdict = gate.evaluate(predictor, current_df, reference_df=ref_df)

    console.print("")
    console.rule("Gate Verification Verdict")
    console.kv("Gate Status", verdict.status)
    console.kv("Schema Check", "PASS" if verdict.schema_ok else "FAIL")
    console.kv("Leakage Check", "PASS" if verdict.leakage_ok else "FAIL")
    console.kv("Drift Check", "PASS" if verdict.drift_ok else "FAIL")

    if verdict.reasons:
        console.print("")
        console.warn("Rejection Reasons:")
        for r in verdict.reasons:
            console.print(f"  • {r}")

    if output_path:
        save_json(output_path, verdict.to_dict())
        console.success(f"Wrote deployment gate verdict JSON -> {output_path}")

    if verdict.passed:
        console.print("")
        console.success("DEPLOYMENT APPROVED: Model and data certified safe for production.")
        return 0
    else:
        console.print("")
        console.error("DEPLOYMENT REJECTED: Model gate check failed.")
        return 1
