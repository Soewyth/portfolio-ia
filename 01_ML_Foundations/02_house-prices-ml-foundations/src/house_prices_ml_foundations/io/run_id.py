from __future__ import annotations

from datetime import datetime


def make_run_id(tag: str = None) -> str:
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{tag}_{run_time}"
    return run_time
