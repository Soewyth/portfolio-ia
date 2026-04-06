from __future__ import annotations  # for future compatibility

import json
from pathlib import Path  # for handling file paths


def save_report_json(path: Path, payload: dict) -> None:
    """Save model evaluation metrics to a JSON report file.
    Args:
        path (Path): Path to save the report file.
        payload (dict): Dictionary containing model evaluation metrics.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
