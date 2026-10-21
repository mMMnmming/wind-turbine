# Stage2 v2：网络学习有向成本与剩余任务

所有 Linux 命令在仓库根目录、原训练环境执行。本页使用独立数据、模型、评估目录，不覆盖 `train-best64` 或原87最优解。不自动运行200轮实验或提交 Git。

## 1. 改了什么，没改什么

- `edge_attention_v2`：7维节点输入，128维、3层、8头编码器；每层用正反向飞行时间生成可学习注意力偏置。动态解码上下文包括全图、未访问节点、当前节点、机巢和剩余预算。最后打分还读取当前点/候选点/机巢之间的有向时间。
- 纯 REINFORCE：每图8条独立**网络采样**路线，共享编码；每条路线与同图另外7条的平均奖励比较（留一法），没有教师、最优标签或训练中求解器。
- 保持原奖励、物理风况、完整巡检/返航和续航判定。阵风仍是固定12 m/s特征，不是随机阵风物理仿真；不恢复最近邻可完成掩码，不加局部搜索。
- 默认128条候选是 **1条贪心＋127条按网络概率采样的路线**，不是127条均匀随机排列。按可行性优先、飞行时间其次选择；全部不可行则保留超限最少路线并标记失败。择优不读取最优参考。
- 增加候选数可以帮助寻找好路线，但不等于网络学得更好。分别报告旧/新网络的贪心，以及相同128预算的结果。网络不保证全局最优，最优证明仍来自分支定界。

默认配置见 `stage2/config_edge_v2.json`：64图/批×8轨迹，100批/轮，200轮；Adam 1e-4按固定200轮余弦衰减至1e-5，梯度裁剪1；策略种子1236、数据种子1234；102真实分区＋128 KDE风场，每5轮替换32个，固定16个独立验证风场。每轮按固定128候选验证，可行率优先、平均归一化飞行时间其次选 `best.pt`，同时记录贪心成绩。只训练一个最终配置，不自动消融或搜索。

## 2. 准备独立训练/验证清单

```bash
python -u -m stage2.run prepare_data --input turbine102.tsv --stage1-config stage1/config.json --config stage2/config_edge_v2.json --data-dir stage2/outputs/edge-v2-data --output stage2/outputs/edge-v2-data
```

清单可以复用，但不能修改；改变风场数、数据种子或风况时换数据目录。训练池刷新不会改写初始清单或验证集。

可选：训练前独立求验证参考（**不是87测试参考**），每区域本次限时60秒：

```bash
mkdir -p "stage2/Branch and Bound/outputs/edge-v2-validation" && nohup python -u "stage2/Branch and Bound/run.py" --regions-json stage2/outputs/edge-v2-data/validation_regions.json --time-limit 60 --output "stage2/Branch and Bound/outputs/edge-v2-validation" > "stage2/Branch and Bound/outputs/edge-v2-validation/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

等该进程结束再读取参考文件。中断/超时后在原命令加入 `--resume` 继续；不能向同一目录同时启动两个求解器。未证明最优的区域不计算真实gap。参考只用于诊断，不决定训练梯度和选模规则。

## 3. 一条命令后台训练

无需验证参考即可运行：

```bash
mkdir -p stage2/outputs/train-edge-v2 && nohup python -u -m stage2.run train --input turbine102.tsv --stage1-config stage1/config.json --config stage2/config_edge_v2.json --data-dir stage2/outputs/edge-v2-data --output stage2/outputs/train-edge-v2 > "stage2/outputs/train-edge-v2/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

如已完成验证参考，训练命令额外加：

```text
--reference-results "stage2/Branch and Bound/outputs/edge-v2-validation/reference_results.json"
```

没有提前准备数据时，训练会自动生成清单。不要给训练传87最优参考；输入指纹不属于验证集时会报错。

恢复（保留同目录 `best.pt` 和 `latest.pt`）：

```bash
mkdir -p stage2/outputs/train-edge-v2 && nohup python -u -m stage2.run resume --checkpoint stage2/outputs/train-edge-v2/latest.pt --config stage2/config_edge_v2.json --data-dir stage2/outputs/edge-v2-data --output stage2/outputs/train-edge-v2 > "stage2/outputs/train-edge-v2/$(date +%Y%m%d_%H%M%S)_resume.log" 2>&1 < /dev/null &
```

可改变目标 `--epochs`，但 `--lr-schedule-epochs` 是固定余弦时钟，恢复不能改变；超过该时钟后保持最小学习率。恢复会检查架构、超参数、清单、环境和数据池，保存/恢复优化器、调度器及随机状态，不支持旧权重跨架构续训。

新诊断包括 `policy_entropy`、`advantage_std`、`unique_route_fraction`、`gradient_clip_fraction` 和 `validation.greedy`；它们位于 `training_history.json`/`metrics.jsonl`。熵低可能是策略集中，也可能过早失去探索，需结合验证成本判断。训练目标是改善采样路线平均奖励，不是直接求128候选最小值的精确梯度。

新增覆盖参数：`--architecture`、`--rollouts-per-instance`、`--decode-strategy greedy|best_of_n`、`--decode-samples`、`--decode-chunk-size`、`--evaluation-seed`、`--lr-min`、`--lr-schedule-epochs`。新版配置至少2轨迹；不要只给旧配置加架构名称而忽略配套训练设置。旧12组 `search` 不支持v2。

## 4. 在同一份87冻结清单测试

先按 `stage2/Branch and Bound/README.md` 完成87独立求解；已有 `optimal87/regions.json` 和参考则直接复用，不重新部署。

```bash
mkdir -p stage2/outputs/model87-edge-v2 && nohup python -u -m stage2.run evaluate --regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json" --checkpoint stage2/outputs/train-edge-v2/best.pt --decode-strategy best_of_n --decode-samples 128 --decode-chunk-size 32 --evaluation-seed 1234 --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --evaluation-output stage2/outputs/model87-edge-v2 > "stage2/outputs/model87-edge-v2/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

- `model_metrics.json`：128候选主结果及总时间；路线图、其他评估产物也对应这个方法，不标成贪心。
- `greedy_metrics.json`：独立执行并计时的单次贪心辅助结果。
- 元数据保存架构、候选数、分块大小、种子和设备；使用实例指纹派生局部采样种子，不消耗训练随机流。同种子/设备/分块可重复，跨硬件不保证位级一致。
- `--random-samples 1000` 保持原意，是额外的纯随机基线数量，不是网络采样数量。
- 主计时包含一次编码、全部候选解码、路线CPU取回、实际成本/可行性检查及择优，CUDA边界同步；不含模型加载、特征/时间矩阵构造、随机基线、绘图、写文件，不加预热。首次执行可能较慢，贪心辅助在主结果之后执行，二者不是受控硬件速度实验。
- 已有评估结果拒绝覆盖，重新测试使用新目录。

旧模型也以同样128候选预算评估：

```bash
python -u -m stage2.run evaluate --regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json" --checkpoint stage2/outputs/train-best64/best.pt --decode-strategy best_of_n --decode-samples 128 --decode-chunk-size 32 --evaluation-seed 1234 --evaluation-output stage2/outputs/model87-old-best128
```

旧checkpoint无需迁移；缺少架构字段默认旧模型，默认贪心逻辑保持。要单独跑贪心可指定 `--decode-strategy greedy`。

## 5. 生成两类gap表

```bash
python "stage2/Branch and Bound/compare.py" --model-results stage2/outputs/model87-edge-v2/model_metrics.json --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --output "stage2/Branch and Bound/outputs/comparison-edge-v2-best128"
```

```bash
python "stage2/Branch and Bound/compare.py" --model-results stage2/outputs/model87-edge-v2/greedy_metrics.json --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --output "stage2/Branch and Bound/outputs/comparison-edge-v2-greedy"
```

旧模型的 `model_metrics.json`/`greedy_metrics.json` 同理生成表。分别对照旧/新贪心、旧/新128候选；不可只将新128结果与旧贪心比较。总gap按有效区域累计飞行时间计算，不平均区域百分比，不把不可行或未证明区域算进去；同时显示全区域原始合计。

原87成绩已被查看，报告应注明，不宣称从未查看的盲测。不要根据87反复挑种子/调参；固定验证集选定后再报告87，未改善也如实保留结果。验证最优参考可用同样比较脚本检查。完整性/正确性测试通过不代表已证明200轮训练更优。

## 测试

```bash
python -m unittest discover -s stage2/tests -v
python -m unittest discover -s "stage2/Branch and Bound/tests" -v
```
