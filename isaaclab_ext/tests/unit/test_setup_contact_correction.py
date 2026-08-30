from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[2] / "scripts/setup_contact_correction.py"
SPEC = importlib.util.spec_from_file_location("setup_contact_correction", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def test_runtime_command_enables_active_contact_correction() -> None:
    command = setup.runtime_command(
        runtime_python="/opt/isaac/python",
        stock_policy=Path("/models/stock.pt"),
        visible_checkpoint=Path("/models/visible.pt"),
        correction_policy=Path("/models/residual.pt"),
        output=Path("/out/run.json"),
        steps=500,
    )

    assert command == [
        "/opt/isaac/python",
        str(setup.RUN_SCRIPT),
        "--mode",
        "active",
        "--stock-policy",
        "/models/stock.pt",
        "--visible-checkpoint",
        "/models/visible.pt",
        "--contact-correction-policy",
        "/models/residual.pt",
        "--output",
        "/out/run.json",
        "--steps",
        "500",
    ]
