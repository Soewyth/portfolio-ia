from __future__ import annotations

import os

from house_prices_ml_foundations.config.paths import get_paths, get_project_root

root_dir = get_project_root()
paths = get_paths(root_dir)
path_mlrun = paths["mlruns"]


DEFAULT_MLFLOW_TRACKING_URI = f"file:{path_mlrun}"
# allow overriding the MLFLOW_TRACKING_URI via env varriable
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
