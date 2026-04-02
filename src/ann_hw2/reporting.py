from __future__ import annotations

from pathlib import Path

import pandas as pd

from ann_hw2.utils import ensure_dir, write_table


EXPERIMENT_LABELS = {
    "resnext50_scratch": "ResNeXt-S",
    "resnext50_finetune": "ResNeXt-FT",
    "densenet121_scratch": "DenseNet-S",
    "densenet121_finetune": "DenseNet-FT",
}
EXPECTED_EXPERIMENTS = list(EXPERIMENT_LABELS)
CLASS_NAME_MAP = {
    "angular_leaf_spot": "角斑病",
    "bean_rust": "锈病",
    "healthy": "健康叶片",
}


def collect_metrics(outputs_root: str | Path) -> pd.DataFrame:
    metrics_paths = sorted(Path(outputs_root).glob("*/metrics.csv"))
    if not metrics_paths:
        raise FileNotFoundError("No metrics.csv files were found in outputs/runs.")
    frames = [pd.read_csv(path) for path in metrics_paths]
    result = pd.concat(frames, ignore_index=True)
    result = result.set_index("experiment_name")
    result = result.reindex(EXPECTED_EXPERIMENTS)
    result.index.name = "experiment_name"
    result = result.reset_index()
    result["experiment_label"] = result["experiment_name"].map(EXPERIMENT_LABELS)
    return result


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value * 100:.2f}"


def _format_minutes(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.1f}"


def _format_integer(value: float) -> str:
    if pd.isna(value):
        return "--"
    return str(int(value))


def _format_million_params(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value / 1_000_000:.2f}"


def write_summary_tables(metrics: pd.DataFrame, latex_dir: str | Path) -> None:
    latex_dir = ensure_dir(latex_dir)

    result_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"实验组别 & Acc(\%) $\uparrow$ & Macro-F1 $\uparrow$ & 最优轮次 & 参数量(M) \\",
        r"\midrule",
    ]
    for row in metrics.itertuples(index=False):
        result_lines.append(
            f"{row.experiment_label} & "
            f"{_format_percent(row.test_accuracy)} & "
            f"{_format_percent(row.test_macro_f1)} & "
            f"{_format_integer(row.best_epoch)} & "
            f"{_format_million_params(row.total_parameters)} \\\\"
        )
    result_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(Path(latex_dir) / "results_table.tex", result_lines)

    time_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"实验组别 & 训练时长(min) & 权重初始化 & 训练方式 \\",
        r"\midrule",
    ]
    for row in metrics.itertuples(index=False):
        if pd.isna(row.training_mode):
            init_type = "--"
            mode_label = "--"
        else:
            init_type = "ImageNet" if row.training_mode == "finetune" else "随机初始化"
            mode_label = "微调" if row.training_mode == "finetune" else "从头训练"
        time_lines.append(
            f"{row.experiment_label} & "
            f"{_format_minutes(row.train_minutes)} & "
            f"{init_type} & "
            f"{mode_label} \\\\"
        )
    time_lines.extend([r"\bottomrule", r"\end{tabular}"])
    write_table(Path(latex_dir) / "time_table.tex", time_lines)


def write_dataset_table(dataset_summary: dict, latex_dir: str | Path) -> None:
    latex_dir = ensure_dir(latex_dir)
    split_sizes = dataset_summary["split_sizes"]
    class_names = [
        CLASS_NAME_MAP.get(name, name.replace("_", " ")) for name in dataset_summary["class_names"]
    ]
    lines = [
        r"\begin{tabular}{p{0.17\columnwidth}p{0.12\columnwidth}p{0.52\columnwidth}}",
        r"\toprule",
        r"数据划分 & 样本数 & 类别说明 \\",
        r"\midrule",
        f"Train & {split_sizes['train']} & {', '.join(class_names)} \\\\",
        f"Validation & {split_sizes['validation']} & 三分类 \\\\",
        f"Test & {split_sizes['test']} & 三分类 \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write_table(Path(latex_dir) / "dataset_table.tex", lines)


def write_result_macros(metrics: pd.DataFrame, latex_dir: str | Path) -> None:
    latex_dir = ensure_dir(latex_dir)
    available = metrics.dropna(subset=["test_accuracy", "test_macro_f1", "train_minutes"], how="all")
    if available.empty:
        lines = [
            r"\newcommand{\BestAccModel}{待生成}",
            r"\newcommand{\BestAccValue}{--}",
            r"\newcommand{\BestFOneModel}{待生成}",
            r"\newcommand{\BestFOneValue}{--}",
            r"\newcommand{\FastestModel}{待生成}",
            r"\newcommand{\FastestTime}{--}",
            r"\newcommand{\SlowestModel}{待生成}",
            r"\newcommand{\SlowestTime}{--}",
        ]
    else:
        best_acc = available.sort_values("test_accuracy", ascending=False).iloc[0]
        best_f1 = available.sort_values("test_macro_f1", ascending=False).iloc[0]
        fastest = available.sort_values("train_minutes", ascending=True).iloc[0]
        slowest = available.sort_values("train_minutes", ascending=False).iloc[0]

        lines = [
            f"\\newcommand{{\\BestAccModel}}{{{EXPERIMENT_LABELS[best_acc['experiment_name']]}}}",
            f"\\newcommand{{\\BestAccValue}}{{{_format_percent(best_acc['test_accuracy'])}\\%}}",
            f"\\newcommand{{\\BestFOneModel}}{{{EXPERIMENT_LABELS[best_f1['experiment_name']]}}}",
            f"\\newcommand{{\\BestFOneValue}}{{{_format_percent(best_f1['test_macro_f1'])}\\%}}",
            f"\\newcommand{{\\FastestModel}}{{{EXPERIMENT_LABELS[fastest['experiment_name']]}}}",
            f"\\newcommand{{\\FastestTime}}{{{_format_minutes(fastest['train_minutes'])}}}",
            f"\\newcommand{{\\SlowestModel}}{{{EXPERIMENT_LABELS[slowest['experiment_name']]}}}",
            f"\\newcommand{{\\SlowestTime}}{{{_format_minutes(slowest['train_minutes'])}}}",
        ]
    write_table(Path(latex_dir) / "result_macros.tex", lines)


def write_report_assets(metrics: pd.DataFrame, latex_dir: str | Path) -> None:
    write_summary_tables(metrics=metrics, latex_dir=latex_dir)
    write_result_macros(metrics=metrics, latex_dir=latex_dir)
