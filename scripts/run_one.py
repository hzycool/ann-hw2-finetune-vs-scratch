from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ann_hw2.train import run_experiment
from ann_hw2.utils import load_yaml, merge_dicts


CONFIG_NAME_MAP = {
    ("resnext50", "scratch"): "resnext50_scratch.yaml",
    ("resnext50", "finetune"): "resnext50_finetune.yaml",
    ("densenet121", "scratch"): "densenet121_scratch.yaml",
    ("densenet121", "finetune"): "densenet121_finetune.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one course-project experiment.")
    parser.add_argument("--model", choices=["resnext50", "densenet121"], required=True)
    parser.add_argument("--mode", choices=["scratch", "finetune"], required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--scratch-epochs", type=int, default=None)
    parser.add_argument("--head-epochs", type=int, default=None)
    parser.add_argument("--full-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = ROOT / "configs"
    base_config = load_yaml(config_dir / "base.yaml")
    run_config = load_yaml(config_dir / CONFIG_NAME_MAP[(args.model, args.mode)])
    config = merge_dicts(base_config, run_config)

    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.scratch_epochs is not None:
        config["training"]["scratch_epochs"] = args.scratch_epochs
    if args.head_epochs is not None:
        config["training"]["head_epochs"] = args.head_epochs
    if args.full_epochs is not None:
        config["training"]["full_epochs"] = args.full_epochs
    if args.device is not None:
        config["training"]["device"] = args.device

    result = run_experiment(config)
    metrics = result["metrics"]
    print(
        f"[done] {metrics['experiment_name']}: "
        f"test_acc={metrics['test_accuracy']:.4f}, "
        f"test_macro_f1={metrics['test_macro_f1']:.4f}, "
        f"train_minutes={metrics['train_minutes']:.2f}"
    )
    print(f"[saved] {result['run_dir']}")


if __name__ == "__main__":
    main()
