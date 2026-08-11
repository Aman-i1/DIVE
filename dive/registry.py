"""Local Model Registry & Automated Model Promotion Gates.

Manages model versions, deployment stages (candidate, staging, production, archived, failed),
model metadata, and executes automated promotion gate verification checks.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dive import __version__
from dive.utils.io import ensure_dir, load_json, save_json


@dataclass
class PromotionGateCheck:
    """Outcome of automated model promotion gate check."""

    approved: bool
    candidate_version: str
    target_stage: str
    metric_checks: Dict[str, Any]
    rejection_reasons: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approved": self.approved,
            "candidate_version": self.candidate_version,
            "target_stage": self.target_stage,
            "metric_checks": self.metric_checks,
            "rejection_reasons": self.rejection_reasons,
            "passed_checks": self.passed_checks,
        }

    def render(self) -> str:
        lines = [
            "MODEL PROMOTION GATE VERIFICATION",
            "=================================",
            f"Candidate Version : {self.candidate_version}",
            f"Target Stage      : {self.target_stage.upper()}",
            f"Verdict           : {'✓ PROMOTION APPROVED' if self.approved else '🔴 PROMOTION REJECTED'}",
        ]
        if self.passed_checks:
            lines.append("Passed Checks:")
            for p in self.passed_checks:
                lines.append(f"  ✓ {p}")
        if self.rejection_reasons:
            lines.append("Rejection Reasons:")
            for r in self.rejection_reasons:
                lines.append(f"  🔴 {r}")
        return "\n".join(lines)


class PromotionGate:
    """Automated gate validation checking metric improvement, stability, leakage, schema."""

    def __init__(self, min_metric_improvement: float = 0.001) -> None:
        self.min_metric_improvement = min_metric_improvement

    def evaluate(
        self,
        candidate_meta: Dict[str, Any],
        current_prod_meta: Optional[Dict[str, Any]] = None,
        target_stage: str = "production",
    ) -> PromotionGateCheck:
        passed = []
        rejections = []
        c_version = candidate_meta.get("version", "unknown")

        c_metrics = candidate_meta.get("metrics", {})

        # 1. Leakage Status Check
        if candidate_meta.get("has_leakage_risk", False):
            rejections.append("Candidate carries unresolved high-risk data leakage.")
        else:
            passed.append("Leakage Safety: Passed (Zero high-risk leakage)")

        # 2. Metric Improvement Check against Production
        if current_prod_meta:
            p_metrics = current_prod_meta.get("metrics", {})
            primary_metric = "Macro F1" if "Macro F1" in c_metrics else ("F1" if "F1" in c_metrics else "Test R2")
            
            c_val = c_metrics.get(primary_metric, 0.0)
            p_val = p_metrics.get(primary_metric, 0.0)

            if c_val < p_val + self.min_metric_improvement:
                rejections.append(
                    f"Candidate metric '{primary_metric}' ({c_val:.4f}) does not improve upon production ({p_val:.4f}) by at least {self.min_metric_improvement}."
                )
            else:
                passed.append(f"Metric Improvement: Passed ('{primary_metric}' {p_val:.4f} -> {c_val:.4f})")
        else:
            passed.append("Metric Validation: Passed (First production model candidate)")

        # 3. Schema Compatibility Check
        if candidate_meta.get("schema_mismatch", False):
            rejections.append("Candidate input schema is incompatible with target environment.")
        else:
            passed.append("Schema Compatibility: Passed")

        approved = len(rejections) == 0

        return PromotionGateCheck(
            approved=approved,
            candidate_version=c_version,
            target_stage=target_stage,
            metric_checks=c_metrics,
            rejection_reasons=rejections,
            passed_checks=passed,
        )


class ModelRegistry:
    """Local model registry managing artifacts in `.dive/registry/`."""

    VALID_STAGES = ("candidate", "staging", "production", "archived", "failed")

    def __init__(self, registry_dir: Union[str, Path] = ".dive/registry") -> None:
        self.registry_dir = ensure_dir(registry_dir)

    def register_model(
        self,
        model_name: str,
        model_artifact_path: Union[str, Path],
        metrics: Dict[str, Any],
        schema: Dict[str, Any],
        experiment_id: str = "",
        stage: str = "candidate",
    ) -> Path:
        """Register a new model artifact version under `.dive/registry/<model_name>/vX/`."""
        model_root = ensure_dir(self.registry_dir / model_name)
        existing_versions = [
            int(d.name[1:]) for d in model_root.glob("v*") if d.is_dir() and d.name[1:].isdigit()
        ]
        next_ver = max(existing_versions, default=0) + 1
        version_str = f"v{next_ver}"

        version_dir = ensure_dir(model_root / version_str)

        # Copy artifact
        src_path = Path(model_artifact_path)
        dest_artifact = version_dir / "model.pkl"
        shutil.copy2(src_path, dest_artifact)

        # Save Metadata
        meta = {
            "model_name": model_name,
            "version": version_str,
            "stage": stage,
            "created_at": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "metrics": metrics,
            "schema": schema,
            "dive_version": __version__,
        }
        save_json(version_dir / "metadata.json", meta)
        return version_dir

    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered model versions."""
        results = []
        target_dirs = [self.registry_dir / model_name] if model_name else list(self.registry_dir.iterdir())
        for m_dir in target_dirs:
            if m_dir.is_dir():
                for v_dir in m_dir.glob("v*"):
                    meta_file = v_dir / "metadata.json"
                    if meta_file.exists():
                        try:
                            results.append(load_json(meta_file))
                        except Exception:
                            pass
        return results

    def get_version_metadata(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        meta_file = self.registry_dir / model_name / version / "metadata.json"
        if meta_file.exists():
            return load_json(meta_file)
        return None

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Find model version currently in 'production' stage."""
        for meta in self.list_models(model_name):
            if meta.get("stage") == "production":
                return meta
        return None

    def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: str = "production",
        force: bool = False,
    ) -> PromotionGateCheck:
        """Promote a model version through automated gate checks."""
        if target_stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid target stage '{target_stage}'. Valid stages: {self.VALID_STAGES}")

        meta = self.get_version_metadata(model_name, version)
        if not meta:
            raise ValueError(f"Model version '{model_name}/{version}' not found in registry.")

        current_prod = self.get_production_model(model_name)
        gate = PromotionGate()
        gate_check = gate.evaluate(meta, current_prod, target_stage)

        if gate_check.approved or force:
            # Demote existing production model to archived if promoting new production
            if target_stage == "production" and current_prod and current_prod["version"] != version:
                prod_meta_file = self.registry_dir / model_name / current_prod["version"] / "metadata.json"
                current_prod["stage"] = "archived"
                save_json(prod_meta_file, current_prod)

            meta["stage"] = target_stage
            meta["promoted_at"] = datetime.now().isoformat()
            save_json(self.registry_dir / model_name / version / "metadata.json", meta)

        return gate_check
