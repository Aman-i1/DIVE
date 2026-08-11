"""CLI Command logic for `dive audit`."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dive.audit import ComplianceAuditor, export_pdf_certificate
from dive.utils.io import load_dataframe, save_json
from dive.utils.logging import Console


def run_audit(
    console: Console,
    data_path: str,
    target: str,
    group_column: Optional[str] = None,
    time_column: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    """Run compliance audit and generate signed ML Reliability Certificate."""
    console.banner("DIVE COMPLIANCE AND RELIABILITY AUDITOR", f"Auditing dataset: {data_path}")

    with console.spinner(f"Running compliance & leakage audit on {data_path}..."):
        df = load_dataframe(data_path)
        dataset_name = Path(data_path).stem
        auditor = ComplianceAuditor(target=target, group_column=group_column, time_column=time_column)
        cert = auditor.audit(df, dataset_name=dataset_name)

    console.print("")
    console.rule("Audit Certificate Result")
    console.kv("Certificate ID", cert.certificate_id)
    console.kv("Audit Verdict", cert.audit_verdict)
    console.kv("Readiness Score", f"{cert.readiness_score:.1f} / 100")
    console.kv("Leakage Status", "CRITICAL LEAKAGE DETECTED" if cert.leakage_found else "CLEAN (NO LEAKAGE)")
    console.kv("SHA-256 Signature", cert.signature_sha256[:24] + "...")
    console.print("")

    if output_path:
        out = Path(output_path)
        if out.suffix.lower() == ".pdf":
            export_pdf_certificate(cert, out, console=console)
        else:
            save_json(out, cert.to_dict())
            console.success(f"Wrote audit certificate JSON -> {out}")
    else:
        json_out = Path("audit_certificate.json")
        save_json(json_out, cert.to_dict())
        console.success(f"Wrote audit certificate JSON -> {json_out}")
