from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ann_hw2.train import run_experiment
from ann_hw2.utils import load_yaml, merge_dicts, resolve_project_paths


OFFICIAL_CONFIG_FILES = [
    "densenet121_finetune.yaml",
    "resnext50_finetune.yaml",
    "densenet121_scratch.yaml",
    "resnext50_scratch.yaml",
]


def main() -> None:
    config_dir = ROOT / "configs"
    outputs_root = ROOT / "outputs" / "runs"
    base_config = load_yaml(config_dir / "base.yaml")

    for config_name in OFFICIAL_CONFIG_FILES:
        run_config = load_yaml(config_dir / config_name)
        config = merge_dicts(base_config, run_config)
        config = resolve_project_paths(config, ROOT)
        run_dir = outputs_root / config["experiment_name"]

        if run_dir.exists():
            shutil.rmtree(run_dir)

        print(f"[start] {config['experiment_name']}")
        try:
            result = run_experiment(config)
        except Exception:
            print(f"[failed] {config['experiment_name']}")
            raise

        metrics = result["metrics"]
        print(
            f"[done] {metrics['experiment_name']}: "
            f"test_acc={metrics['test_accuracy']:.4f}, "
            f"test_macro_f1={metrics['test_macro_f1']:.4f}, "
            f"train_minutes={metrics['train_minutes']:.2f}"
        )


if __name__ == "__main__":
    main()
