from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_sandbox_cli_streams_hsic_phases_without_corrupting_result_envelope(
    tmp_path: Path,
) -> None:
    source = """
import openturns as ot
model = ot.SymbolicFunction(["private_x"], ["private_x * private_x"])
problem = ot.Uniform(-1.0, 1.0)
"""
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "source": source,
                "seed": 42,
                "analysis": {
                    "analysis_key": "hsic",
                    "config": {"sample_size": 50, "permutations": 5},
                    "output_targets": [0],
                },
            }
        )
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "services.compute.cli",
            "execute",
            str(request_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(completed.stdout)
    events = [
        json.loads(line.removeprefix("UNCERTAINTYCAT_PROGRESS "))
        for line in completed.stderr.splitlines()
        if line.startswith("UNCERTAINTYCAT_PROGRESS ")
    ]

    assert envelope["status"] == 200
    assert envelope["body"]["result"]["analysis_key"] == "hsic"
    assert [event["phase"] for event in events] == [
        "model_loading",
        "applicability",
        "openturns",
        "sampling",
        "kernel_construction",
        "observed_indices",
        "permutation_inference",
        "ranking",
        "serializing",
    ]
    assert [event["percent"] for event in events] == sorted(event["percent"] for event in events)
    assert (
        next(event for event in events if event["phase"] == "permutation_inference")[
            "indeterminate"
        ]
        is True
    )
    assert "private_x" not in completed.stderr
    assert "SymbolicFunction" not in completed.stderr
