"""Fail closed when an analysis-plugin change lacks adversarial scientific evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import openturns as ot

from uncertaintycat_core.catalog import get_plugin

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIRECTORY = ROOT / "docs" / "openturns-sync" / "evidence"
PLUGIN_DIRECTORY = ROOT / "uncertaintycat_core" / "plugins"
PINNED_UPSTREAM_URL = re.compile(
    r"^https://github\.com/openturns/openturns/(?:blob|commit)/[0-9a-f]{40}/?"
)
PLUGIN_EXCLUSIONS = {"__init__.py", "base.py"}
REQUIRED_SYNC_FILES = {
    "docs/SCIENTIFIC_VALIDATION.md",
    "docs/openturns-sync/README.md",
    "docs/openturns-sync/state.json",
}


class PolicyError(ValueError):
    """Raised when retained scientific evidence does not satisfy policy."""


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{label} must be an object")
    return value


def _string(value: object, label: str, *, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise PolicyError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: object, label: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise PolicyError(f"{label} must contain at least {minimum} item(s)")
    return [_string(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _python_test_path(node_id: str, label: str) -> Path:
    parts = node_id.split("::")
    if len(parts) < 2 or not parts[-1].startswith("test_"):
        raise PolicyError(f"{label} must be a pytest node ID ending in a test function")
    relative = Path(parts[0])
    if not relative.as_posix().startswith("tests/") or relative.suffix != ".py":
        raise PolicyError(f"{label} must point to a Python test under tests/")
    path = ROOT / relative
    if not path.is_file():
        raise PolicyError(f"{label} points to missing file {relative.as_posix()}")
    function_name = parts[-1].split("[")[0]
    if not re.search(rf"^def {re.escape(function_name)}\s*\(", path.read_text(), re.MULTILINE):
        raise PolicyError(f"{label} points to missing function {function_name}")
    return relative


def _browser_test_path(reference: str, label: str) -> Path:
    if "#" not in reference:
        raise PolicyError(f"{label} must use path#test-title syntax")
    path_text, title = reference.split("#", 1)
    relative = Path(path_text)
    if not relative.as_posix().startswith("apps/web/e2e/") or relative.suffix != ".ts":
        raise PolicyError(f"{label} must point to a Playwright test under apps/web/e2e/")
    path = ROOT / relative
    if not path.is_file():
        raise PolicyError(f"{label} points to missing file {relative.as_posix()}")
    if _string(title, label) not in path.read_text():
        raise PolicyError(f"{label} title is not present in {relative.as_posix()}")
    return relative


def validate_manifest(path: Path) -> tuple[str, list[str], set[str]]:
    """Validate one manifest and return its key, declared pytest nodes and evidence files."""

    try:
        document = _mapping(json.loads(path.read_text()), path.as_posix())
    except json.JSONDecodeError as exc:
        raise PolicyError(f"{path.as_posix()} is not valid JSON: {exc}") from exc

    if document.get("schemaVersion") != "1.0.0":
        raise PolicyError(f"{path.as_posix()} must use schemaVersion 1.0.0")
    plugin_key = _string(document.get("pluginKey"), "pluginKey")
    if path.stem != plugin_key:
        raise PolicyError(f"{path.name} must describe pluginKey {path.stem!r}")
    plugin = get_plugin(plugin_key)
    if _string(document.get("pluginVersion"), "pluginVersion") != plugin.version:
        raise PolicyError(f"{plugin_key}: manifest pluginVersion does not match the catalog")
    if (
        _string(document.get("resultSchemaVersion"), "resultSchemaVersion")
        != plugin.result_schema_version
    ):
        raise PolicyError(f"{plugin_key}: manifest resultSchemaVersion does not match the catalog")
    if _string(document.get("openturnsVersion"), "openturnsVersion") != ot.__version__:
        raise PolicyError(
            f"{plugin_key}: manifest OpenTURNS version does not match the installed pin"
        )

    upstream = document.get("upstreamEvidence")
    if not isinstance(upstream, list) or len(upstream) < 2:
        raise PolicyError(
            f"{plugin_key}: upstreamEvidence needs implementation and benchmark sources"
        )
    upstream_kinds: set[str] = set()
    for index, item in enumerate(upstream):
        evidence = _mapping(item, f"upstreamEvidence[{index}]")
        kind = _string(evidence.get("kind"), f"upstreamEvidence[{index}].kind")
        url = _string(evidence.get("url"), f"upstreamEvidence[{index}].url")
        if not PINNED_UPSTREAM_URL.match(url):
            raise PolicyError(
                f"{plugin_key}: upstream source {url!r} must pin an exact 40-character "
                "OpenTURNS commit"
            )
        upstream_kinds.add(kind)
    if not {"implementation", "benchmark"}.issubset(upstream_kinds):
        raise PolicyError(
            f"{plugin_key}: upstreamEvidence must include implementation and benchmark"
        )

    benchmark_tests = _string_list(document.get("benchmarkTests"), "benchmarkTests")
    applicability_tests = _string_list(document.get("applicabilityTests"), "applicabilityTests")
    resource = _mapping(document.get("resourceModel"), "resourceModel")
    if resource.get("bounded") is not True:
        raise PolicyError(f"{plugin_key}: resourceModel.bounded must be true")
    _string(resource.get("complexity"), "resourceModel.complexity", minimum=4)
    _string(resource.get("formula"), "resourceModel.formula", minimum=4)
    _string(resource.get("unit"), "resourceModel.unit", minimum=4)
    _string(resource.get("upstreamDerivation"), "resourceModel.upstreamDerivation", minimum=40)
    oracle_test = _string(
        resource.get("independentOracleTest"), "resourceModel.independentOracleTest"
    )
    boundary_tests = _string_list(
        resource.get("boundaryTests"), "resourceModel.boundaryTests", minimum=2
    )
    audit_answers = _mapping(resource.get("auditAnswers"), "resourceModel.auditAnswers")
    for question in ("hiddenLoopNesting", "maximumConfiguration", "uiMaximumDimension"):
        _string(audit_answers.get(question), f"resourceModel.auditAnswers.{question}", minimum=40)

    interpretation_boundaries = _string_list(
        document.get("interpretationBoundaries"), "interpretationBoundaries", minimum=2
    )
    if any(len(item) < 20 for item in interpretation_boundaries):
        raise PolicyError(f"{plugin_key}: interpretation boundaries must be explicit sentences")
    browser_tests = _string_list(document.get("browserTests"), "browserTests")

    python_nodes = [*benchmark_tests, *applicability_tests, oracle_test, *boundary_tests]
    evidence_files = {
        _python_test_path(node_id, f"declared test {node_id}").as_posix()
        for node_id in python_nodes
    }
    evidence_files.update(
        _browser_test_path(reference, f"declared browser test {reference}").as_posix()
        for reference in browser_tests
    )
    return plugin_key, list(dict.fromkeys(python_nodes)), evidence_files


def changed_paths(base: str, head: str) -> dict[str, str]:
    result = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", base, head, "--"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    changes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        status = fields[0][0]
        path = fields[-1]
        changes[path] = status
    return changes


def enforce_changed_plugin_policy(
    changes: dict[str, str], manifests: dict[str, tuple[Path, set[str]]]
) -> None:
    changed = set(changes)
    plugin_changes = {
        path
        for path in changed
        if path.startswith("uncertaintycat_core/plugins/")
        and Path(path).name not in PLUGIN_EXCLUSIONS
        and path.endswith(".py")
    }
    if not plugin_changes:
        return
    missing_sync = REQUIRED_SYNC_FILES - changed
    if missing_sync:
        raise PolicyError(
            "Plugin changes must refresh scientific validation, sync guidance and state: "
            + ", ".join(sorted(missing_sync))
        )
    if any(changes[path] in {"A", "D"} for path in plugin_changes) and (
        "uncertaintycat_core/catalog.py" not in changed
    ):
        raise PolicyError("Added or deleted plugins must update catalog registration")

    for plugin_path in sorted(plugin_changes):
        plugin_key = Path(plugin_path).stem
        manifest_relative = f"docs/openturns-sync/evidence/{plugin_key}.json"
        status = changes[plugin_path]
        if status == "D":
            if manifest_relative not in changes or changes[manifest_relative] != "D":
                raise PolicyError(
                    f"Deleted plugin {plugin_key} must delete its scientific evidence manifest"
                )
            continue
        if plugin_key not in manifests:
            raise PolicyError(f"Changed plugin {plugin_key} has no scientific evidence manifest")
        if manifest_relative not in changed:
            raise PolicyError(
                f"Changed plugin {plugin_key} must refresh {manifest_relative} in the same change"
            )
        _, evidence_files = manifests[plugin_key]
        changed_evidence = evidence_files & changed
        python_evidence = {path for path in changed_evidence if path.endswith(".py")}
        browser_evidence = {path for path in changed_evidence if path.endswith(".ts")}
        if not python_evidence:
            raise PolicyError(
                f"Changed plugin {plugin_key} must change a declared Python evidence test"
            )
        if not browser_evidence:
            raise PolicyError(
                f"Changed plugin {plugin_key} must change a declared browser contract test"
            )


def run_declared_tests(node_ids: Iterable[str]) -> None:
    unique = list(dict.fromkeys(node_ids))
    if not unique:
        return
    subprocess.run([sys.executable, "-m", "pytest", *unique], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base Git revision for diff-aware enforcement")
    parser.add_argument(
        "--head", default="HEAD", help="Head Git revision for diff-aware enforcement"
    )
    parser.add_argument("--run-declared-tests", action="store_true")
    arguments = parser.parse_args()
    if bool(arguments.base) is False and arguments.head != "HEAD":
        parser.error("--head requires --base")

    manifest_paths = sorted(EVIDENCE_DIRECTORY.glob("*.json"))
    if not manifest_paths:
        raise SystemExit("No OpenTURNS scientific evidence manifests were found")
    manifests: dict[str, tuple[Path, set[str]]] = {}
    declared_tests: list[str] = []
    try:
        for path in manifest_paths:
            plugin_key, tests, evidence_files = validate_manifest(path)
            manifests[plugin_key] = (path, evidence_files)
            declared_tests.extend(tests)
        if arguments.base:
            enforce_changed_plugin_policy(changed_paths(arguments.base, arguments.head), manifests)
        if arguments.run_declared_tests:
            run_declared_tests(declared_tests)
    except (PolicyError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Scientific-change policy failed: {exc}") from exc
    print(
        f"Scientific-change policy passed for {len(manifests)} retained manifest(s) "
        f"and {len(set(declared_tests))} declared Python test(s)."
    )


if __name__ == "__main__":
    main()
