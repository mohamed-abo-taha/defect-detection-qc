"""An ILLUSTRATIVE labor-offset model for defect-QC automation.

This is a transparent what-if calculator, NOT a guarantee or a sales claim. Every input
is an assumption *you* provide; the output explicitly nets the cost of false alarms
(crying wolf wastes inspector time) and missed defects (escapes that reach customers)
against the labor saved. The article's "$X saved" figures should come from a model like
this with your numbers stated — never asserted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass
class ROIInputs:
    parts_per_day: int = 20_000
    seconds_manual_inspection: float = 10.0
    inspector_cost_per_hour: float = 30.0
    defect_rate: float = 0.03
    model_defect_recall: float = 0.95       # fraction of true defects the model catches
    model_false_alarm_rate: float = 0.02    # fraction of good parts the model wrongly flags
    human_review_fraction: float = 1.0      # share of model-flagged parts a human re-checks
    cost_per_escaped_defect: float = 20.0   # cost when a missed defect ships
    working_days_per_year: int = 250


def estimate(i: ROIInputs) -> dict:
    """Return a per-day / per-year labor-offset estimate (all values are assumptions in -> range out)."""
    baseline_hours = i.parts_per_day * i.seconds_manual_inspection / 3600
    baseline_labor = baseline_hours * i.inspector_cost_per_hour

    defects = i.parts_per_day * i.defect_rate
    good = i.parts_per_day - defects
    flagged = defects * i.model_defect_recall + good * i.model_false_alarm_rate
    review_hours = flagged * i.human_review_fraction * i.seconds_manual_inspection / 3600
    model_labor = review_hours * i.inspector_cost_per_hour

    escapes = defects * (1 - i.model_defect_recall)
    escape_cost = escapes * i.cost_per_escaped_defect

    labor_saved = baseline_labor - model_labor
    net = labor_saved - escape_cost
    return {
        "baseline_labor_per_day": baseline_labor,
        "model_labor_per_day": model_labor,
        "labor_saved_per_day": labor_saved,
        "escaped_defects_per_day": escapes,
        "escape_cost_per_day": escape_cost,
        "net_saving_per_day": net,
        "net_saving_per_year": net * i.working_days_per_year,
    }


def main():
    ap = argparse.ArgumentParser(description="Illustrative QC-automation ROI model")
    for k, v in asdict(ROIInputs()).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    inp = ROIInputs(**{k: getattr(args, k) for k in asdict(ROIInputs())})
    print("ASSUMPTIONS (yours to change):")
    for k, v in asdict(inp).items():
        print(f"  {k:28s} {v}")
    print("\nESTIMATE (illustrative — not a guarantee):")
    for k, v in estimate(inp).items():
        print(f"  {k:28s} ${v:,.0f}" if "defect" not in k or "cost" in k else f"  {k:28s} {v:,.1f}")


if __name__ == "__main__":
    main()
