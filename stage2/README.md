# Stage 2：服务器完整实验运行指南

本目录实现论文第三阶段的风场感知无人机巡检路径规划。下面给出在 Linux CUDA 服务器上从数据准备、训练到评估的命令。命令均从仓库根目录执行。

> 重要说明：这些命令运行的是当前仓库实现的完整实验流程。论文没有公开完整网络超参数、训练细节或预训练 checkpoint；当前实现与论文在网络结构、状态/奖励细节和训练数据生成上存在差异。因此，实验结果应称为“基于论文方法的实现与评估”，不能仅凭运行成功称为严格复现论文全部结果。

## 1. 获取项目并确认数据

在服务器上克隆仓库（将占位符替换成你的 Git 仓库地址）：

```bash
git clone <仓库地址> wind-uav
cd wind-uav
```

如果仓库没有远程地址，也可以把整个项目目录上传到服务器。训练和默认评估需要仓库根目录中的 `turbine102.tsv`、`turbine87.tsv`，以及 `stage1/config.json`。先确认文件存在：

```bash
ls -l turbine102.tsv turbine87.tsv stage1/config.json
```

`turbine102.tsv` 用于训练数据构造；`turbine87.tsv` 用于评估。数据格式为每行一个风机 ID、X 坐标和 Y 坐标，制表符分隔，坐标单位为 km。

## 2. 创建 Python 环境并安装依赖

示例使用 Python 3.10 或 3.11：

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

CUDA 12.1 环境按仓库锁定版本安装：

```bash
python -m pip install -r stage2/requirements-cuda121.txt
```

该依赖文件安装 PyTorch 2.5.1 CUDA 12.1 wheel 和其余依赖。服务器 NVIDIA 驱动需能支持该 CUDA wheel。其他 CUDA 版本请先从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择匹配的 PyTorch 安装命令，再安装通用依赖：

```bash
python -m pip install -r stage2/requirements.txt
```

确认 PyTorch 能识别 GPU：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

默认设备是 `cuda`；没有可用 CUDA 时完整训练会报错。`--device cpu` 仅建议用于小规模调试，不建议用 CPU 跑默认完整训练。

## 3. 准备训练数据

默认配置在 `stage2/config.json`：随机种子 1234、32 个 KDE 虚拟风场、隐藏维度 128、3 层编码器、8 个注意力头、batch size 512、100 epochs、每 epoch 100 batches、学习率 `1e-4`。虚拟风场由 `turbine102.tsv` 拟合 KDE 后生成，再经过 Stage 1 分区流程。

创建本次实验的独立输出目录并生成训练区域：

```bash
mkdir -p stage2/outputs/experiment/train
cp stage2/config.json stage2/outputs/experiment/config.json
python -m stage2.run prepare_data \
  --input turbine102.tsv \
  --stage1-config stage1/config.json \
  --config stage2/config.json \
  --output stage2/outputs/experiment/train
```

命令会打印生成的 Stage 1 路由区域数量，并保存 `training_regions.json` 和 `input_hashes.json`。默认 32 个虚拟风场的准备过程会重复运行 Stage 1，可能需要一些时间。`--virtual-fields 0` 可禁用 KDE 数据，只适合快速调试，不是默认完整配置。

## 4. 运行完整训练

```bash
python -m stage2.run train \
  --input turbine102.tsv \
  --stage1-config stage1/config.json \
  --config stage2/config.json \
  --output stage2/outputs/experiment/train \
  2>&1 | tee stage2/outputs/experiment/train/train.log
```

默认设置为 100 epochs × 100 batches/epoch，batch size 为 512。训练会逐 epoch 保存 `latest.pt`，并在训练指标改善时更新 `best.pt`；同时写入 `metrics.jsonl`、`training_history.json` 和 `config_snapshot.json`。`best.pt` 用于最终评估，`latest.pt` 用于断点续训。

如果训练中断，从最近 checkpoint 继续。`--epochs` 是最终目标 epoch 总数，不是额外增加的轮数：

```bash
python -m stage2.run resume \
  --input turbine102.tsv \
  --stage1-config stage1/config.json \
  --config stage2/config.json \
  --checkpoint stage2/outputs/experiment/train/latest.pt \
  --epochs 100 \
  --output stage2/outputs/experiment/train \
  2>&1 | tee -a stage2/outputs/experiment/train/train.log
```

例如要从已有 100 轮 checkpoint 接着训练到 120 轮，将 `--epochs` 改为 `120`。

## 5. 在 87 台风机数据上评估

评估使用 `best.pt`，执行模型贪心解码、定向最近邻基线和固定随机种子的随机路线对照：

```bash
mkdir -p stage2/outputs/experiment/evaluation_run
python -m stage2.run evaluate \
  --input turbine87.tsv \
  --stage1-config stage1/config.json \
  --config stage2/config.json \
  --checkpoint stage2/outputs/experiment/train/best.pt \
  --output stage2/outputs/experiment/evaluation_run \
  --evaluation-output stage2/outputs/experiment/evaluation_run/turbine87_best \
  --random-samples 1000 \
  --evaluation-seed 1234 \
  --plot-dpi 300 \
  2>&1 | tee stage2/outputs/experiment/evaluation_run/evaluate.log
```

评估产物包括 `evaluation.log`、`evaluation_summary.json`、`evaluation_routes.json`、`evaluation_regions.csv`、成本对比图和各区域路线图。**评估时必须使用独立的 `--output` 目录**：当前 CLI 会在所有模式下写入 `<output>/training_regions.json` 和 `<output>/input_hashes.json`。如果评估和训练共用同一个 `--output`，评估输入会覆盖训练数据清单与输入哈希，降低训练记录的可审计性。

当前仓库在 `turbine87.tsv` 上生成的 Stage 1 测试分区为 4 个区域；论文正文描述 Walney 3 的 8 个区域。因此，即使上述评估全部成功，也应报告实际生成的区域数量，并说明它与论文测试设置不一致。

## 6. 检查、归档和下载结果

检查训练是否达到目标轮数、checkpoint 是否存在，以及评估区域数和可行率：

```bash
tail -n 5 stage2/outputs/experiment/train/metrics.jsonl
ls -lh stage2/outputs/experiment/train/best.pt stage2/outputs/experiment/train/latest.pt
cat stage2/outputs/experiment/evaluation_run/turbine87_best/evaluation_summary.json
cat stage2/outputs/experiment/evaluation_run/turbine87_best/evaluation.log
```

保存代码版本和输入数据校验值，便于复核：

```bash
git rev-parse HEAD > stage2/outputs/experiment/git_commit.txt
sha256sum turbine102.tsv turbine87.tsv stage1/config.json > stage2/outputs/experiment/input_sha256.txt
sha256sum stage2/outputs/experiment/config.json >> stage2/outputs/experiment/input_sha256.txt
```

在本地机器上从服务器下载完整结果（替换用户名、主机和仓库路径）：

```bash
scp -r <用户名>@<服务器地址>:<仓库路径>/stage2/outputs/experiment ./stage2-experiment
```

## 输出结构

```text
stage2/outputs/experiment/
├── config.json
├── train/
│   ├── best.pt
│   ├── latest.pt
│   ├── config_snapshot.json
│   ├── input_hashes.json
│   ├── metrics.jsonl
│   ├── training_history.json
│   └── training_regions.json
└── evaluation_run/
    ├── evaluate.log
    ├── input_hashes.json
    ├── training_regions.json
    └── turbine87_best/
        ├── evaluation.log
        ├── evaluation_summary.json
        ├── evaluation_routes.json
        ├── evaluation_regions.csv
        ├── cost_comparison.png
        ├── model_routes_overview.png
        └── region_*_routes.png
```

## 实验边界

风向相关飞行时间、巢出发闭环路径、悬停等效时间、续航可行性掩码和 REINFORCE baseline 均已在代码中实现。网络结构、部分状态/奖励、KDE 虚拟数据设置及基线方法是当前仓库的实现选择；这些细节不能从论文中完整恢复。报告实验时应同时给出代码版本、配置、数据哈希、训练曲线、实际测试区域数和评估汇总，不要将默认配置描述为论文原文公布的超参数。
