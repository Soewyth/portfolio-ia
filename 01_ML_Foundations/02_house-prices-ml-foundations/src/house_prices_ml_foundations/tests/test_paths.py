from __future__ import annotations

from pathlib import Path

import pytest

from house_prices_ml_foundations.config.paths import get_paths, get_project_root


def test_get_project_root() -> None:
    """Test that get_project_root returns the project root."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_get_paths_creates_outputs_but_not_datasets_raw(tmp_path: Path) -> None:
    """get_paths should create output dirs, but raise if datasets/raw is missing."""
    root_dir = tmp_path
    (root_dir / "datasets").mkdir()

    with pytest.raises(FileNotFoundError, match="datasets/raw"):
        get_paths(root_dir)

    # datasets/raw must not be created automatically
    assert not (root_dir / "datasets" / "raw").exists()

    # output directories are allowed to be created
    assert (root_dir / "outputs").exists()
    assert (root_dir / "outputs" / "figures").exists()
    assert (root_dir / "outputs" / "mlruns").exists()
    assert (root_dir / "outputs" / "models").exists()
    assert (root_dir / "outputs" / "reports").exists()
    assert (root_dir / "outputs" / "submissions").exists()


def test_get_paths_returns_paths_when_datasets_raw_exists(tmp_path: Path) -> None:
    """get_paths should return all expected paths when datasets/raw exists."""
    root_dir = tmp_path
    (root_dir / "datasets" / "raw").mkdir(parents=True)

    paths = get_paths(root_dir)

    assert paths["datasets_raw"] == root_dir / "datasets" / "raw"
    assert paths["outputs"] == root_dir / "outputs"
    assert paths["reports"] == root_dir / "outputs" / "reports"
    assert paths["models"] == root_dir / "outputs" / "models"
    assert paths["submissions"] == root_dir / "outputs" / "submissions"
    assert paths["figures"] == root_dir / "outputs" / "figures"
    assert paths["mlruns"] == root_dir / "outputs" / "mlruns"
