from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import get_settings
from app.services.explainability_service import ExplainabilityService


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate explainability evidence artifacts from saved formal outputs.")
    parser.add_argument("--artifact-prefix", default="seed99_robustness")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--output-prefix", default="seed99")
    args = parser.parse_args()

    service = ExplainabilityService(get_settings())
    try:
        result = service.generate_explainability_artifacts(
            artifact_prefix=args.artifact_prefix,
            output_prefix=args.output_prefix,
            sample_size=args.sample_size,
            allow_overwrite=True,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    summary = result["summary"]
    warnings = summary["warnings"]
    print(f"Users included: {summary['run_context']['users_included']}")
    print(f"Recommendations explained: {summary['run_context']['recommendations_explained']}")
    print(f"Summary: {service.settings.explainability_summary_path(args.output_prefix)}")
    print(f"Audit: {service.settings.explainability_audit_path(args.output_prefix)}")
    print(f"Examples: {service.settings.explainability_examples_csv_path(args.output_prefix)}")
    print(f"Rank shift: {service.settings.rank_shift_analysis_csv_path(args.output_prefix)}")
    print(f"Case studies: {service.settings.explainability_case_studies_path(args.output_prefix)}")
    print("Warnings:")
    for warning in warnings:
        print(f"- {warning}")
    print("Missing optional fields:")
    for field in result["audit"]["missing_optional_fields"]:
        print(f"- {field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
