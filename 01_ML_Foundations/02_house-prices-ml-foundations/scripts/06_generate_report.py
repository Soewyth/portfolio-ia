from __future__ import annotations

from house_prices_ml_foundations.config.paths import get_paths, get_project_root
from house_prices_ml_foundations.evaluation.report_md import generate_report_md


def main() -> None:
    """Generate consolidated markdown audit report from latest JSON artifacts."""
    root_dir = get_project_root()
    paths = get_paths(root_dir)

    report_path = generate_report_md(paths["reports"])

    print("=== Markdown report generated ===")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
