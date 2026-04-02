# 人工神经网络课程作业：从头训练 V.S. 微调

本项目完成“报告二：利用深度学习框架对比微调和从头训练；从头训练 V.S. 微调”的课程实验。实验选择 `ResNeXt50_32x4d` 和 `DenseNet121` 两种卷积神经网络，在 `Beans` 三分类图像数据集上分别进行从头训练与基于 ImageNet 预训练权重的微调，对比四组实验在准确率、Macro-F1、训练时间和收敛速度上的差异。

## 1. 环境依赖

- Python 3.12
- PyTorch 2.7.0
- torchvision 0.22.0
- datasets 2.21.0
- scikit-learn
- matplotlib
- pandas
- XeLaTeX / latexmk / biber

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 2. 项目结构

```text
.
|-- configs/                # 四组实验配置
|-- report/                 # LaTeX 报告源码
|-- scripts/                # 训练与结果汇总脚本
`-- src/ann_hw2/            # 数据、模型、训练与可视化模块
```

## 3. 数据准备

默认使用 Hugging Face `beans` 数据集。首次运行训练脚本时会自动下载并缓存到 `data/beans_cache/`。数据集包含 `train / validation / test` 三个划分，无需手动整理目录。

## 4. 训练命令

单组实验：

```bash
python scripts/run_one.py --model resnext50 --mode scratch
python scripts/run_one.py --model resnext50 --mode finetune
python scripts/run_one.py --model densenet121 --mode scratch
python scripts/run_one.py --model densenet121 --mode finetune
```

顺序运行全部四组实验：

```bash
python scripts/run_all.py
```

如果需要做快速连通性检查，可以临时缩短 epoch：

```bash
python scripts/run_one.py --model resnext50 --mode finetune --head-epochs 1 --full-epochs 1
```

## 5. 说明

本仓库主要保留实验代码、配置文件与运行入口，便于直接复现实验过程。训练产生的权重、日志和图像结果默认不纳入代码仓库，可按需要在本地保存。
