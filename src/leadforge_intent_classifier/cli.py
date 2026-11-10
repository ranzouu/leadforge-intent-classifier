from __future__ import annotations

import os
from pathlib import Path

from leadforge_intent_classifier.experiment import run_experiment, save_runs


def main() -> int:
    demo = os.getenv("DEMO_MODE", "0") == "1"
    mode = "demo" if demo else "trained"
    print(f"Rodando intent classifier em modo={mode}...")
    records, summary = run_experiment(demo=demo)
    latest_path, summary_path = save_runs(records, summary)
    print(f"Salvo: {latest_path}")
    print(f"Resumo: {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0
