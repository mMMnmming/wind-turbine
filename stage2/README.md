# Stage 2：数据、训练、搜索与测试

Stage2 在 102 机组数据上构造训练拓扑，使用独立的 KDE 拓扑验证并选模，最后可单独在 87 数据集评估。所有命令从仓库根目录运行。本实现保留现有 Transformer+GRU 主干；论文未公开的奖励权重和训练细节按本目录配置执行，应作为复现假设报告。

## 环境

建议 Python 3.10+。CUDA 12.1 示例：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r stage2/requirements-cuda121.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

无 CUDA 环境仅用于小型调试时指定 `--device cpu`。确认仓库根目录包含 `turbine102.tsv`、`turbine87.tsv` 和 `stage1/config.json`。

## 训练数据及风况

```bash
python -m stage2.run prepare_data \
  --input turbine102.tsv --stage1-config stage1/config.json \
  --config stage2/config.json --output stage2/outputs/experiment
```

训练集 = 102 数据经 Stage1 得到的真实区域 + 配置指定的 KDE 随机拓扑；验证集为同一空间分布用独立种子生成的 16 个拓扑场。输出分别为 `training_regions.json`、`validation_regions.json` 和 `data_manifest.json`。Stage1 不会被修改。平均风速取 Stage1 配置值（论文设定为 6 m/s），设计阵风默认为 12 m/s，作为状态特征；不生成或假设阵风概率分布。风向采用论文罗盘角约定（0°北、90°东）。

部分 KDE 随机拓扑可能无法通过 Stage1 最终分配的可行性检查。Stage2 会保留原约束，使用确定性的新种子重采样该拓扑，直到得到指定数量的有效拓扑；每个场最多尝试 `kde_max_attempts_per_field` 次（默认 20 次），耗尽后明确报错。失败原因会写入控制台/`nohup` 日志，所有尝试的种子、接受状态和区域数量写入 `data_manifest.json` 的 `kde_generation`。因此合成集是以 Stage1 部署可行为条件的 KDE 样本，训练集和验证集都采用这一筛选规则。原始 102/87 数据若部署失败仍直接报错。

## 单配置训练

```bash
python -m stage2.run train \
  --input turbine102.tsv --stage1-config stage1/config.json \
  --config stage2/config.json --output stage2/outputs/experiment/train \
  2>&1 | tee stage2/outputs/experiment/train.log
```

每个训练 batch 由不同区域/随机拓扑组成，变长图通过 padding 和有效节点掩码处理；若实例少于请求 batch size，则使用所有不同实例，不重复复制图。动作掩码仅排除机巢和已访问节点，完整巡检后隐式返回机巢；路线若超出续航，以大额安全惩罚处理。每个 epoch 在固定验证集贪心评估，优先按可行率、再按归一化飞行时间选 `best.pt`。`latest.pt` 用于续训，`best.pt` 用于评估。

奖励是归一化航程/飞行时间成本（权重 1）、归一化能耗成本（初始权重 0.5）、可行完整巡检终端奖励（+5）、续航违规惩罚（−20，另加超限惩罚）。这些是工程起点，不是论文公布的精确系数。可通过 config 或命令行覆盖训练超参数；当前支持 `--learning-rate`、`--energy-weight`、`--virtual-fields`、`--epochs`、`--batch-size` 和 `--batches-per-epoch`。

续训示例（`--epochs` 是最终总 epoch 数）：

```bash
python -m stage2.run resume \
  --input turbine102.tsv --stage1-config stage1/config.json \
  --config stage2/config.json --checkpoint stage2/outputs/experiment/train/latest.pt \
  --epochs 120 --output stage2/outputs/experiment/train
```

## 12 组超参数搜索

```bash
python -m stage2.run search \
  --input turbine102.tsv --stage1-config stage1/config.json \
  --config stage2/config.json --output stage2/outputs/search
```

搜索网格为学习率 `{3e-5, 1e-4, 3e-4}` × KDE 训练场数 `{32,64}` × 能耗权重 `{0.25,0.5}`，共 12 组。验证排名前三的配置再各用 3 个种子复跑，按三种子平均可行率及平均归一化飞行时间选最终 checkpoint。所有尝试保存在 `search/` 下，汇总为 `search/search_results.json`，最终模型为 `search/final/best.pt`。完整搜索约训练 21 次完整模型，计算量显著高于单次训练；调试时可用较小的 `--epochs`、`--batches-per-epoch` 和 `--batch-size`。

## 87 数据集最终评估

仅在确定模型和超参数后运行测试；测试结果不会参与选模或搜索。将 `--checkpoint` 指向单次训练的 `best.pt` 或搜索的 `search/final/best.pt`：

```bash
python -m stage2.run evaluate \
  --input turbine87.tsv --stage1-config stage1/config.json \
  --config stage2/config.json --checkpoint stage2/outputs/search/search/final/best.pt \
  --output stage2/outputs/test \
  --evaluation-output stage2/outputs/test/turbine87_final \
  --random-samples 1000 --evaluation-seed 1234
```

评估只写入 `evaluation-output` 下的测试清单和结果，不会覆盖训练/验证清单。产物包括 `test_regions.json`、`evaluation_summary.json`、路线和 CSV 明细、最近邻比较及图表。汇总记录输入哈希、checkpoint 配置、可行率、飞行时间/等效能耗、航程及模型相对最近邻的飞行时间变化。

当前 Stage1 对 `turbine87.tsv` 产生的区域数量可能与论文报告的 Walney 3 八个区域不同；应报告实际数量，不能称为完全复现了论文测试分区。
