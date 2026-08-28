from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TARGET_HSIC_MANIFEST = ROOT / "docs" / "openturns-sync" / "evidence" / "target_hsic.json"
POLICY_SPEC = importlib.util.spec_from_file_location(
    "check_scientific_change", ROOT / "scripts" / "check_scientific_change.py"
)
assert POLICY_SPEC and POLICY_SPEC.loader
POLICY = importlib.util.module_from_spec(POLICY_SPEC)
POLICY_SPEC.loader.exec_module(POLICY)
PolicyError = POLICY.PolicyError
enforce_changed_plugin_policy = POLICY.enforce_changed_plugin_policy
validate_manifest = POLICY.validate_manifest


def test_retained_target_hsic_manifest_matches_catalog_and_declared_evidence() -> None:
    plugin_key, tests, evidence_files = validate_manifest(TARGET_HSIC_MANIFEST)

    assert plugin_key == "target_hsic"
    assert "tests/core/test_resource_envelopes.py" in evidence_files
    assert "apps/web/e2e/ui-flows.spec.ts" in evidence_files
    assert any("independent_loop_oracle" in test for test in tests)


def test_manifest_rejects_unpinned_upstream_evidence(tmp_path: Path) -> None:
    document = json.loads(TARGET_HSIC_MANIFEST.read_text())
    document["upstreamEvidence"][0]["url"] = (
        "https://github.com/openturns/openturns/blob/master/example.cxx"
    )
    manifest = tmp_path / "target_hsic.json"
    manifest.write_text(json.dumps(document))

    with pytest.raises(PolicyError, match="exact 40-character"):
        validate_manifest(manifest)


def test_changed_plugin_requires_refreshed_python_and_browser_evidence() -> None:
    manifest = Path("docs/openturns-sync/evidence/target_hsic.json")
    evidence_files = {
        "tests/core/test_resource_envelopes.py",
        "apps/web/e2e/ui-flows.spec.ts",
    }
    base_changes = {
        "uncertaintycat_core/plugins/target_hsic.py": "M",
        manifest.as_posix(): "M",
        "docs/SCIENTIFIC_VALIDATION.md": "M",
        "docs/openturns-sync/README.md": "M",
        "docs/openturns-sync/state.json": "M",
    }

    with pytest.raises(PolicyError, match="declared Python evidence"):
        enforce_changed_plugin_policy(
            base_changes,
            {"target_hsic": (manifest, evidence_files)},
        )

    with pytest.raises(PolicyError, match="declared browser contract"):
        enforce_changed_plugin_policy(
            {**base_changes, "tests/core/test_resource_envelopes.py": "M"},
            {"target_hsic": (manifest, evidence_files)},
        )

    enforce_changed_plugin_policy(
        {
            **base_changes,
            "tests/core/test_resource_envelopes.py": "M",
            "apps/web/e2e/ui-flows.spec.ts": "M",
        },
        {"target_hsic": (manifest, evidence_files)},
    )
