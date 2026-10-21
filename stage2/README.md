# Stage 2：数据、训练、搜索与测试

Stage2 在 102 机组数据上构造训练拓扑，使用独立的 KDE 拓扑验证并选模，最后可单独在 87 数据集评估。所有命令从仓库根目录运行。

**新版训练请看 [有向成本注意力＋同图多轨迹强化学习](README_edge_v2.md)**：使用独立的 `config_edge_v2.json`，从头训练，不覆盖旧模型；默认128条网络候选择优，并另外保存单次贪心成绩。它是算法改进，不宣称逐项复现原论文网络。

已训练好的新版模型可按 [两组微调说明](README_finetune.md) 保留旧权重，运行普通微调与熵正则微调各50轮，分别保存贪心/128候选最佳模型。

本页以下是旧版 `config.json` 的 Transformer+GRU 流程，继续兼容。论文未公开的奖励权重和训练细节应作为复现假设报告。

注意力打分已修复为按编码后的隐藏维度缩放，而不是按 7 维输入特征缩放；网络参数结构、奖励和动作约束不变。旧 checkpoint 仍可加载，但采样概率及后续训练会变化。对照实验请使用新的输出目录从头训练，不要将新旧版本的训练记录混为一次实验。

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

## 本轮推荐：固定评测 + 可刷新的训练池

本轮仅改数据覆盖、抽样及评测/恢复机制；不更换 Transformer+GRU、REINFORCE、冻结贪心 baseline、奖励和动作约束，不使用教师或最优路线监督。默认配置为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `learning_rate` | 0.0001 | 本次旧版固定数据搜索选出的学习率 |
| `virtual_fields` | 64 | 本次搜索选出的初始 KDE 风场数，每场仍由原 Stage1 分区 |
| `energy_weight` | 0.25 | 本次搜索选出的能耗权重 |
| `validation_fields` | 16 | 固定验证风场数，不随训练刷新 |
| `data_seed` | 1234 | 数据生成与刷新种子，独立于策略训练种子 |
| `seed` | 1234 | 网络初始化、路线采样、batch 抽样种子 |
| `refresh_every_epochs` | 5 | 在第 6、11、16…轮开始前刷新 |
| `refresh_fields` | 16 | 每次替换最旧的 16 个完整虚拟风场，真实分区不移除 |
| `stratified_sampling` | true | 按每 10 个待巡检节点的规模桶均衡抽样，不复制同图 |

学习率/初始风场数/能耗权重采用本次搜索胜出值，但不保证在刷新训练中仍然最优。刷新量随初始池从 256 场调整到 64 场而由 64 改为 16，保留每次替换 25% 的节奏；16 不是搜索得出的参数。刷新与分层抽样仍是工程设定，不是论文原始超参数。固定同源 KDE 验证只能衡量同分布泛化，不能代替分布外测试。当前阵风仍是固定 12 m/s 输入，不是动态阵风仿真。

默认训练种子仍为 1234，数据种子仍为 1234。搜索中最终保留模型来自种子 1236，不代表 1236 是普遍最佳的超参数；若想用该种子重新训练，可加 `--seed 1236`，不要因此改动 `--data-seed`。若要按旧搜索的固定数据条件复跑，另加 `--refresh-every-epochs 0 --no-stratified-sampling`；新代码修复等差异仍可能影响结果。

### 服务器一条命令后台训练（日期日志）

先激活已经能运行原 Stage2 的 Python/CUDA 环境，在仓库根目录执行：

```bash
mkdir -p stage2/outputs/round1 && nohup python -u -m stage2.run train --output stage2/outputs/round1 > "stage2/outputs/round1/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

首次自动准备 64+16 个风场，再训练 100 epoch。数据准备也在后台，日志会显示进度；不要将此前旧实验（尤其 256 场缓存）的目录当作新的 `round1`。训练不需要安装 Gurobi，不会自动测试 87，也不会自动执行 21 次超参数搜索。可用 `tail -f stage2/outputs/round1/具体日期.log` 查看进度。不要同时向同一训练输出目录启动多个任务。

正式训练前可先做一次很小的 CUDA 流程检查（不代表模型质量）：

```bash
python -m stage2.run train --output stage2/outputs/round1_smoke --virtual-fields 2 --validation-fields 1 --refresh-fields 1 --refresh-every-epochs 1 --epochs 2 --batches-per-epoch 1 --batch-size 2
```

若显存不足，正式命令可以增加 `--batch-size 128` 或 `--batch-size 64`。降低批量会改变每轮采样路线数，比较实验时请同时比较累计采样量和耗时。

### 对照与多种子复跑

关闭刷新及分层抽样、恢复 32 场数据，建立修复后的旧方法对照：

```bash
python -m stage2.run train --virtual-fields 32 --refresh-every-epochs 0 --no-stratified-sampling --output stage2/outputs/baseline32
```

也可以保留默认 64 场而只加 `--refresh-every-epochs 0`，单独比较刷新作用。以上对照使用相同 `data_seed` 和验证场数，验证拓扑相同。建议逐项修改，而不是把数据量、抽样和训练预算一起改变后归因于单一因素。

多种子复跑可复用已经准备的数据（先等首次准备完成），只修改 `--seed`，不修改 `--data-seed`：

```bash
python -m stage2.run train --data-dir stage2/outputs/round1 --seed 1235 --output stage2/outputs/round1_seed1235
```

### 数据和续训如何保存

`--data-dir` 默认为 `--output`。其中 `training_regions.json` 是**初始**训练清单，`validation_regions.json` 是固定验证清单，`validation_baselines.json` 保存最近邻标尺；`data_manifest.json` 保存参数、哈希及初次生成记录。重复使用会检查参数和文件哈希，拒绝静默重建或覆盖。源 TSV/配置哈希只忽略 Windows/Linux 换行差异，准备好的 JSON 文件按原始字节检查。

训练输出的 `pool_history/pool_0000.json`、`pool_0001.json`…保存实际活跃数据池、刷新种子和重采样记录；不会覆盖初始清单。`metrics.jsonl` 记录数据池版本、每轮采样路线数、训练/验证可行率、最近邻相对改进率和分规模验证指标。`best.pt` 的选择规则仍是“可行率优先，归一化飞行时间其次”，新增参考 gap 不改变选模。

checkpoint 还保存数据池和 Python/NumPy/PyTorch 随机状态。续训保留原参数，只增加最终总 epoch 数：

```bash
mkdir -p stage2/outputs/round1 && nohup python -u -m stage2.run resume --checkpoint stage2/outputs/round1/latest.pt --output stage2/outputs/round1 --epochs 150 > "stage2/outputs/round1/$(date +%Y%m%d_%H%M%S)_resume.log" 2>&1 < /dev/null &
```

如果原训练用了单独的 `--data-dir`、非默认 seed/batch 或刷新设置，续训时也必须提供相同值；并保留原 `best.pt`。同环境下已用 CPU 小实验验证逐 tensor 一致；不承诺跨设备或跨 PyTorch 版本的位级一致。旧 checkpoint 没有数据池状态，不能直接接入默认的刷新训练；请开新实验。checkpoint 通过 pickle 加载，只加载你自己生成/信任的文件。

如需连续记录参考 gap，续训时也传入原 `--reference-results`。迁移完整实验时请保留初始数据清单、整个 `pool_history` 和 `best.pt`/`latest.pt`，而不只是一个模型文件。

### 可选：本地 Gurobi 离线标尺

只对固定验证子集求解，不为每个训练 batch 求解，也不把参考路线用于损失函数。需要可用的 Gurobi 命令行和有效许可证，**不要求 `gurobipy`**。

1. 可在本地先准备数据，再将整个数据目录原样复制到服务器；服务器的源 TSV/Stage1 配置须通过哈希检查：

```bash
python -m stage2.run prepare_data --output stage2/outputs/round1_data
```

2. 在安装并授权 Gurobi 的机器求解固定、按节点规模分位数抽取的最多 32 个验证区域：

```bash
python -m stage2.benchmark --regions-json stage2/outputs/round1_data/validation_regions.json --output stage2/outputs/round1_reference --max-instances 32 --time-limit 120 --threads 4
```

Windows 找不到命令时可附加 `--gurobi-command "C:/gurobi1103/win64/bin/gurobi_cl.exe"`。模型直接使用有向飞行时间、全访问、入/出度、MTZ 子环消除和同一等效续航约束，不做矩阵对称化。`--time-limit` 是每个区域的求解预算，不保证在预算内证明最优。默认请求 `MIPGap=0`；结果保存状态、目标值、下界、gap、路线和求解日志/版本。

`OPTIMAL` 状态本身只表示达到求解器设定的容差。代码将“容差内最优”和“目标上下界数值闭合”分开记录：只有后者进入 `mean_optimality_gap`；超时可行解或尚未闭合的界只进入 `mean_best_known_gap`。无可行解、模型预测不安全、零参考目标值均不伪造 gap；统计同时给出实际比较样本数。这是固定部署后的区域最优，不是部署与路径联合全局最优。[Gurobi 状态说明](https://docs.gurobi.com/projects/optimizer/en/current/reference/numericcodes/statuscodes.html)

完成的区域缓存在 `reference_results.json`，以相同参数重跑会跳过；失败但留下模型/日志的区域不会被覆盖，应使用新的输出目录。求解中断不会把不完整结果伪装成最优解。

3. 将 `reference_results.json` 一并复制到服务器后，可在训练命令中增加：

```bash
python -m stage2.run train --data-dir stage2/outputs/round1_data --reference-results stage2/outputs/round1_reference/reference_results.json --output stage2/outputs/round1_with_reference
```

也可以先正常训练，事后只评估固定验证集，不重训：

```bash
python -m stage2.run evaluate --regions-json stage2/outputs/round1_data/validation_regions.json --checkpoint stage2/outputs/round1_with_reference/best.pt --reference-results stage2/outputs/round1_reference/reference_results.json --evaluation-output stage2/outputs/round1_validation --random-samples 100
```

如果先前训练在 `round1` 中，改用其中的 `best.pt` 及对应 `validation_regions.json`。没有有效许可证时直接跳过整个 Gurobi 步骤，仍可训练并比较最近邻；不要因此声称已经测到了最优性差距。

### 模型与配置确定后，才运行 87 最终测试

```bash
python -m stage2.run evaluate --input turbine87.tsv --checkpoint stage2/outputs/round1/best.pt --evaluation-output stage2/outputs/round1_turbine87 --random-samples 1000
```

不要拿 87 的结果回头选择本轮配置。此前已经查看过 87，最终研究还应增加从未用于开发的独立测试集。本轮不新增推理多采样、局部搜索、多 rollout 或动态状态输入。

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

奖励是归一化航程/飞行时间成本（权重 1）、归一化能耗成本（默认权重 0.25）、可行完整巡检终端奖励（+5）、续航违规惩罚（−20，另加超限惩罚）。能耗权重采用本次搜索胜出值；其余系数仍是工程设定，不是论文公布的精确原值。可通过 config 或命令行覆盖训练超参数；当前支持 `--learning-rate`、`--energy-weight`、`--virtual-fields`、`--epochs`、`--batch-size` 和 `--batches-per-epoch`。

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

这是保留的旧版固定数据搜索入口，会显式关闭刷新与分层抽样，也不接收 `--reference-results`。请使用独立的新输出目录；复用数据缓存时，场数、种子和源文件等设定必须一致，不能将旧 256 场缓存作为 64 场缓存使用。本轮推荐先运行上面的单配置对照，不默认启动这项昂贵搜索。

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

## 回归检查

```bash
python -m unittest discover -s stage2/tests -v
```

覆盖数据缓存/哈希、训练验证隔离、整场刷新、规模抽样、连续训练与续训一致性、CLI 训练/评估和旧搜索调度、求解状态及失败不覆盖。求解器接口的单元测试使用模拟响应；它们不能替代有效许可证下的真实 Gurobi 求解验收。微型 CUDA 检查也不代表模型性能已提升，需要完整对照实验验证。
