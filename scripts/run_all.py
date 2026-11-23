#!/usr/bin/env python3
"""Generate the portfolio, train and evaluate the early-warning models, and write outputs/.

    python scripts/run_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ews.report import run  # noqa: E402

if __name__ == "__main__":
    r = run()
    print(r["summary"].round(3).to_string())
    print(f"\nPSI {r['psi']:.3f}; {len(r['alerts'])} alerts in the latest month. See outputs/ews_report.md")
