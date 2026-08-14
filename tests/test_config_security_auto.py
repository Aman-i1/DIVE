"""Unit & Integration tests for Declarative Config, Security Auditor & dive auto CLI."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from click.testing import CliRunner

from dive.cli import cli
from dive.config import DiveConfig
from dive.security import SecurityAuditResult, SecurityAuditor


def test_declarative_config_io(tmp_path: Path) -> None:
    cfg_data = {
        "target": "churn",
        "mode": "competition",
        "output_dir": str(tmp_path / "out"),
        "validation": {
            "strategy": "StratifiedGroupKFold",
            "cv_splits": 5,
            "group_column": "cust_id",
        },
        "resources": {
            "time_budget_secs": 1200.0,
            "memory_budget_mb": 4096.0,
        },
    }

    json_file = tmp_path / "dive.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f)

    loaded = DiveConfig.load(json_file)
    assert loaded.target == "churn"
    assert loaded.mode == "competition"
    assert loaded.validation.strategy == "StratifiedGroupKFold"
    assert loaded.validation.group_column == "cust_id"
    assert loaded.resources.time_budget_secs == 1200.0


def test_security_path_traversal(tmp_path: Path) -> None:
    base_dir = tmp_path / "sandbox"
    base_dir.mkdir()

    # Valid subpath
    valid = SecurityAuditor.safe_path_join(base_dir, "models/model.pkl")
    assert valid == (base_dir / "models" / "model.pkl").resolve()

    # Path traversal attack
    with pytest.raises(ValueError, match="Path traversal detected"):
        SecurityAuditor.safe_path_join(base_dir, "../../etc/passwd")


def test_security_pickle_audit() -> None:
    # Malicious bytecode payload attempting os.system
    malicious_data = b"cos\nsystem\n(S'echo hacked'\ntR."
    res = SecurityAuditor.audit_pickle_bytes(malicious_data)

    assert isinstance(res, SecurityAuditResult)
    assert res.is_secure is False
    assert res.risk_level == "CRITICAL_RISK"
    assert any("os.system" in w for w in res.warnings)


def test_dive_auto_cli_execution(tmp_path: Path) -> None:
    runner = CliRunner()
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("age,income,churn\n25,30000,0\n35,60000,1\n45,90000,0\n55,120000,1\n65,150000,0\n75,180000,1\n", encoding="utf-8")

    out_dir = tmp_path / "auto_out"

    result = runner.invoke(
        cli,
        [
            "auto",
            str(csv_file),
            "--target",
            "churn",
            "--mode",
            "fast",
            "--budget",
            "60s",
            "--output",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "DIVE INDUSTRIAL AUTONOMOUS ENGINE" in result.output
    assert "Autonomous AutoML execution finished successfully." in result.output
