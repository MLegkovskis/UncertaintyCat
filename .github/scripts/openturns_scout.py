"""Compare the pinned OpenTURNS version with the latest stable PyPI release."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"


def main() -> None:
    source = PYPROJECT.read_text()
    match = re.search(r'"openturns==([^";]+)"', source)
    if not match:
        raise SystemExit("Could not find an exact OpenTURNS pin in pyproject.toml")
    current = match.group(1)
    request = urllib.request.Request(
        "https://pypi.org/pypi/openturns/json",
        headers={"User-Agent": "UncertaintyCat release scout"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        latest = json.load(response)["info"]["version"]
    update = current != latest
    summary = (
        f"# OpenTURNS release scout\n\n"
        f"- Pinned: `{current}`\n- Latest stable PyPI release: `{latest}`\n"
        f"- Update available: `{str(update).lower()}`\n"
    )
    Path(os.environ.get("GITHUB_STEP_SUMMARY", "/dev/null")).write_text(summary)
    if update:
        (ROOT / "openturns-update.md").write_text(
            f"A weekly dependency scan found OpenTURNS `{latest}` while "
            f"UncertaintyCat pins `{current}`.\n\n"
            "Acceptance checklist:\n\n"
            "- [ ] Read the upstream release notes and identify new or changed algorithms.\n"
            "- [ ] Update the pin and lockfile on a branch.\n"
            "- [ ] Run all reference models and scientific regression tests.\n"
            "- [ ] Compare catalog/config/result schemas and add plugins deliberately.\n"
            "- [ ] Record numerical drift and migration notes before merging.\n\n"
            f"PyPI: https://pypi.org/project/openturns/{latest}/\n"
            "Upstream releases: https://github.com/openturns/openturns/releases\n"
        )
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a") as stream:
            stream.write(f"current_version={current}\nlatest_version={latest}\n")
            stream.write(f"update_available={str(update).lower()}\n")


if __name__ == "__main__":
    main()
