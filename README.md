# 人工神经网络课程作业：从头训练 V.S. 微调

本仓库保留课程实验复现所需的代码与配置。实验以 `ResNeXt50_32x4d` 和 `DenseNet121` 为对象，在 `Beans` 三分类数据集上比较从头训练与基于 ImageNet 预训练权重的微调效果。

## 1. 环境依赖

- Python 3.12
- PyTorch 2.7.0
- torchvision 0.22.0
- datasets 2.21.0
- scikit-learn
- matplotlib
- pandas
- PyYAML
- tqdm
- Pillow

安装方式：

```bash
python -m pip install -r requirements.txt
```

## 2. 项目结构

```text
.
|-- configs/          # 实验配置
|-- scripts/          # 训练脚本入口
`-- src/ann_hw2/      # 数据、模型、训练与工具函数
```

## 3. 数据准备

默认使用 Hugging Face `beans` 数据集。首次运行训练脚本时会自动下载并缓存到 `data/beans_cache/`。数据集已包含 `train / validation / test` 三个划分，无需手动整理目录。

## 4. 运行方式

单组实验：

```bash
python scripts/run_one.py --model densenet121 --mode finetune
python scripts/run_one.py --model resnext50 --mode finetune
python scripts/run_one.py --model densenet121 --mode scratch
python scripts/run_one.py --model resnext50 --mode scratch
```

按正式实验顺序连续运行四组实验：

```bash
python scripts/run_official.py
```

正式实验顺序为：

1. `densenet121_finetune`
2. `resnext50_finetune`
3. `densenet121_scratch`
4. `resnext50_scratch`

## 5. 说明

- 默认训练设置为：
  - `DenseNet121 / ResNeXt50 finetune`: `head_epochs=1`, `full_epochs=1`
  - `DenseNet121 / ResNeXt50 scratch`: `scratch_epochs=4`
- 训练过程中生成的权重、日志、曲线图和混淆矩阵默认保存在本地 `outputs/` 目录，不纳入代码仓库。
- 本仓库仅保留实验复现所需代码，便于直接查看和运行。
