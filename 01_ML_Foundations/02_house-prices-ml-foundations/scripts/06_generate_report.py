from __future__ import annotations

from pathlib import Path

from house_prices_ml_foundations.evaluation.report_md import generate_report_md


def main() -> None:
    """Generate consolidated markdown audit report from latest JSON artifacts."""
    scripts_dir = Path(__file__).resolve().parent
    root_dir = scripts_dir.parent


    # Define output directories,
    outputs_path = root_dir / "outputs"
    reports_path = outputs_path / "reports"

    # creating paths
    outputs_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    report_path = generate_report_md(reports_path)

    print("=== Markdown report generated ===")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
