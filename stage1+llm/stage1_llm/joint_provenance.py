"""Identify archived source bytes without trusting a caller's filename/label."""

import hashlib
import json
from pathlib import Path


def source_provenance(source_path):
    source = Path(source_path).read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    manifest = (
        Path(__file__).resolve().parents[1] / "baselines/joint-v5/provenance.json"
    )
    records = (
        json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    )
    for name, record in records.items():
        if record.get("normalized_sha256") == digest:
            return {
                "origin": record["origin"],
                "archive_name": name,
                "source_run": record.get("source_run"),
                "normalized_sha256": digest,
            }
    return {"origin": "unverified_external", "normalized_sha256": digest}
