from __future__ import annotations

import shutil
import sys
from pathlib import Path

import json

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ann_hw2.reporting import collect_metrics, write_dataset_table, write_report_assets
from ann_hw2.utils import ensure_dir


def main() -> None:
    outputs_root = ROOT / "outputs" / "runs"
    summary_dir = ensure_dir(ROOT / "outputs" / "summary")
    latex_dir = ensure_dir(ROOT / "report" / "generated")

    metrics = collect_metrics(outputs_root)
    metrics.to_csv(summary_dir / "summary_metrics.csv", index=False)

    first_dataset_summary = next(outputs_root.glob("*/dataset_summary.json"), None)
    if first_dataset_summary is None:
        raise FileNotFoundError("No dataset_summary.json file was found in outputs/runs.")
    with first_dataset_summary.open("r", encoding="utf-8") as handle:
        dataset_summary = json.load(handle)

    write_report_assets(metrics=metrics, latex_dir=latex_dir)
    write_dataset_table(dataset_summary=dataset_summary, latex_dir=latex_dir)

    plot_targets = {
        "resnext50_scratch_curve.png": outputs_root / "resnext50_scratch" / "training_curves.png",
        "resnext50_finetune_curve.png": outputs_root / "resnext50_finetune" / "training_curves.png",
        "densenet121_scratch_curve.png": outputs_root / "densenet121_scratch" / "training_curves.png",
        "densenet121_finetune_curve.png": outputs_root / "densenet121_finetune" / "training_curves.png",
        "resnext50_finetune_cm.png": outputs_root / "resnext50_finetune" / "confusion_matrix.png",
        "densenet121_finetune_cm.png": outputs_root / "densenet121_finetune" / "confusion_matrix.png",
    }
    for target_name, source_path in plot_targets.items():
        if source_path.exists():
            shutil.copy2(source_path, summary_dir / target_name)
            shutil.copy2(source_path, latex_dir / target_name)

    print(f"[saved] summary CSV: {summary_dir / 'summary_metrics.csv'}")
    print(f"[saved] latex assets: {latex_dir}")


if __name__ == "__main__":
    main()
