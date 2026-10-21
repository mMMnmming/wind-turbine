# 87 数据集：独立最优路线与模型对比

分支定界不读取模型、不等待训练、不重新部署。它读取 **Stage1 已保存的87部署结果 + 配套原始 TSV**，冻结共同测试清单，独立求解并保存路线、图和时间。之后每个模型只测试这份清单，复用同一份最优标尺。

这里只优化固定分区下每个机巢的完整巡检路线，不优化 Stage1 部署。区域相互独立时，各区域最优飞行时间之和就是该固定部署下的累计飞行成本最优；不是部署与路径联合最优，也不是并行无人机的整场完工时间。

所有命令在仓库根目录执行，先激活原有 Python 3.10+ 环境。带空格路径必须保留引号。分支定界无需 PyTorch、GPU 或 Gurobi；绘图需要现有 Matplotlib。模型测试仍需要原来的 PyTorch 环境。

## 1. 准备87的 Stage1结果（只做一次）

如果已有87的 Stage1摘要及当时使用的 TSV，直接跳到第2步，并替换为实际摘要路径。没有时执行：

```bash
MPLBACKEND=Agg python stage1/run.py --input turbine87.tsv --config stage1/config.json --output stage1/results87 --no-show
```

生成 `stage1/results87/summary.json`。不要将默认 `stage1/results/summary.json` 当成87：本地原有默认结果是189台机组、8个机巢，不能与87混用。已有实验目录不要覆盖；要更换部署配置时使用新的 Stage1结果目录和新的最优解目录。

当前 Stage1 对87可能产生4个区域，而不是论文的8个；本工具保留实际结果，不调整分区凑数。

## 2. 后台独立求最优解（不需要任何模型文件）

等待 Stage1结束后执行：

```bash
mkdir -p "stage2/Branch and Bound/outputs/optimal87" && nohup python -u "stage2/Branch and Bound/run.py" --stage1-summary stage1/results87/summary.json --turbines turbine87.tsv --output "stage2/Branch and Bound/outputs/optimal87" > "stage2/Branch and Bound/outputs/optimal87/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

默认不限时，处理全部区域直到证明最优或无解。已有结果时拒绝覆盖；新实验换输出目录。未指定输入时，默认读取 `stage1/results87/summary.json` 和 `turbine87.tsv`，不再默认依赖网络测试目录。也可显式用 `--regions-json 已有区域清单.json` 替代原生摘要；不能与 `--stage1-summary` 同时传入。

### 固定输入与来源

首次校验摘要与 TSV 的节点标识、归属、机巢坐标、配置、服务半径以及已保存的保守路线审计值，然后导出不可覆盖的 `regions.json`，保存实际坐标、节点顺序、机巢、完整风况/续航配置及所传源文件哈希。普通风机坐标来自配套 TSV，机巢与分区来自摘要，绝不重新运行部署。

**来源限制：旧 Stage1摘要没有记录原始 TSV哈希或全部坐标。** 一致性检查不能数学上证明传入 TSV就是历史运行时的原文件；务必提供当时的同源数据。新保存的哈希用于固定本次选定输入并检查后续恢复一致性，不冒充历史来源证明。源文本哈希只忽略 Windows/Linux 换行差异，冻结清单自身按实际字节检查。

平均风速、风向和续航沿用摘要配置；固定阵风特征12 m/s是现有复现设定，不是随机阵风仿真。

### 保存哪些文件

```text
stage2/Branch and Bound/outputs/optimal87/
  regions.json                     共同的固定测试输入，之后模型读取它
  reference_results.json           每区路线、成本、计时、上下界及证明状态
  summary.json                     全部区域飞行/运行时间合计及完整性状态
  results.csv                      每区数值明细
  region_01_optimal_route.png       已证明最优区域的路线图
  region_XX_best_known_route.png    未证明最优时已有的最好路线
  region_XX_no_route_route.png      没有可行路线时的节点与状态图
  routes_overview.png              全部区域总览，未完成区域也标明状态
  checkpoints/                     可恢复的搜索状态
  日期.log                         控制台日志
```

图中标出机巢、风机节点编号/标识、飞行方向、公里坐标、飞行时间和证明状态。未证明的路线不会标成最优。恢复时图随当前结果重画；目录中可能保留早期 `best_known` 图，当前状态以 JSON及总览为准。

查看汇总：

```bash
python -m json.tool "stage2/Branch and Bound/outputs/optimal87/summary.json"
```

重点字段：

- `total_flight_seconds`：所有区域都有路线时的飞行时间合计；没有路线的区域存在时为 null，而不是伪造完整合计。
- `available_flight_seconds`、`route_region_count`：已有路线的部分合计和数量。
- `total_runtime_seconds`：已记录区域的求解运行时间合计，恢复时累计；结合 `reported_count` 和 `runtime_total_complete` 看是否覆盖全部区域。
- `all_regions_proven_optimal`、`flight_total_status`：区分完整最优、完整但仅为最好已知路线、部分结果。
- `proven_optimal_count`、`infeasible_count`、`unfinished_count`：证明完成情况。

### 日志与恢复

```bash
tail -f "$(ls -t 'stage2/Branch and Bound/outputs/optimal87/'*.log | head -n 1)"
```

日志中的 `best` 是当前最好可行路线成本，`lower_bound` 是下界；日志 `gap` 是求解器自己的上下界间隙，不是网络与最优解的 gap。接近零也必须看最终 `OPTIMAL` 状态，不能把超时当最优。

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--time-limit` | 0 | 每区域本次启动的秒数预算，0不限时 |
| `--log-interval` | 10 | 周期日志间隔/秒；发现更好路线与结束时立即输出 |
| `--checkpoint-interval` | 60 | 搜索状态保存间隔/秒，正常中断也保存 |
| `--plot-dpi` | 150 | 路线图分辨率 |
| `--tolerance` | 1e-8 | 绝对最优性容差/秒，只能调小 |
| `--resume` | 关闭 | 恢复原输入及搜索，跳过已完成区域 |

中断后用原命令加 `--resume`：

```bash
mkdir -p "stage2/Branch and Bound/outputs/optimal87" && nohup python -u "stage2/Branch and Bound/run.py" --stage1-summary stage1/results87/summary.json --turbines turbine87.tsv --output "stage2/Branch and Bound/outputs/optimal87" --resume > "stage2/Branch and Bound/outputs/optimal87/$(date +%Y%m%d_%H%M%S)_resume.log" 2>&1 < /dev/null &
```

也可在恢复时传 `--regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json"`，不依赖原始文件继续存在。保留整个最优解目录，尤其冻结输入、参考结果和检查点。不同时向同一目录启动两个任务。SIGINT/SIGTERM正常中断可保存；强杀或断电只能恢复到上次完整检查点。

读取已完成缓存保留原求解时间，不把文件读取时间当成算法时间。旧版本无冻结清单的求解目录不自动迁移；仍可读取旧参考文件比较，但新独立流程请使用新输出目录。

## 3. 模型训练结束后，在同一份87清单上测试

```bash
mkdir -p stage2/outputs/model87-v1 && nohup python -u -m stage2.run evaluate --regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json" --checkpoint stage2/outputs/train-best64/best.pt --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --evaluation-output stage2/outputs/model87-v1 --random-samples 1000 --evaluation-seed 1234 > "stage2/outputs/model87-v1/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

替换为你实际的 `best.pt`。以后换模型只修改 checkpoint 和模型输出目录，**不重新部署，不重求最优解**。不要改用 `--input turbine87.tsv` 来重新生成另一套分区。

显式区域清单的环境为权威输入，不能被 checkpoint静默改写；有环境元数据但不匹配时报错。旧 checkpoint没有该元数据时，会明确警告并保留输入配置，不能虚构已核验一致。六维旧权重不兼容当前七维含阵风模型，不强行迁移。

模型输出原有路线图、CSV和JSON，并新增/补充：

```text
model_metrics.json        每区飞行时间、推理时间、可行性、路线及指纹
  summary:
    total_flight_seconds
    total_runtime_seconds
    feasible_region_count
    infeasible_region_count
evaluation_summary.json  同样包含 model_totals 汇总
```

模型原始飞行时间合计包含超续航路线的实际时间，`infeasible_region_count` 会明确告知，不能当成完整可行巡检成绩。零任务区域直接返回机巢，推理时间记0并不调用网络。所有时间单位为秒。

## 4. 读取两份保存结果生成对比表

等待上述两步完成后执行：

```bash
python "stage2/Branch and Bound/compare.py" --model-results stage2/outputs/model87-v1/model_metrics.json --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --output "stage2/Branch and Bound/outputs/comparison-v1"
```

比较脚本不重新运行模型或求解器。控制台和 Markdown表包含每区域：

| 区域 | 节点数 | 模型飞行时间/s | 分支定界飞行时间/s | 模型运行时间/s | 分支定界运行时间/s | 飞行时间 gap/% | 状态 |
|---|---|---|---|---|---|---|---|

表后清楚区分：

1. **全部区域原始合计**：两种算法全部已测区域的飞行和运行时间，包含不可行/未证明路线，明确缺失、不可行和未证明数量。没有完整路线覆盖时，全场飞行合计为 N/A，另列已有路线合计。
2. **有效对比区域合计**：仅使用“模型可行 + 参考已证明最优 + 参考飞行时间大于0”的相同区域，注明“有效 k / 总 K个区域”。

```text
区域gap = (模型飞行时间 - 最优飞行时间) / 最优飞行时间 × 100%

主汇总gap = (有效区域模型飞行时间合计 - 对应最优飞行时间合计)
            / 对应最优飞行时间合计 × 100%
```

不是区域百分比的简单平均，也不拿运行时间计算这个 gap。5%表示对应模型飞行时间比最优多5%。不可行模型路线、未证明参考和零成本参考的区域 gap为N/A；部分区域的主汇总必须写“部分区域”，不冒充整个87测试的成绩。

输出：

- `comparison.csv`：中文列名、UTF-8 BOM，Excel可读；最后两行分别是原始合计和有效合计，`row_type` 区分区域/原始合计/有效合计。
- `comparison.md`：可读表格和两组汇总。
- `comparison.json`：未四舍五入数值，原始合计在 `summary.model_all_regions` 与 `summary.bnb_all_regions`，主gap在 `summary.flight_gap_percent`。
- 控制台同步打印相同口径。已有表格不覆盖，换模型请用新比较目录。

按有序坐标、机巢和完整环境配置的指纹匹配，不按第几个区域硬配。缺失区域明确警告；重复或伪造最优性证明拒绝；模型明显优于声称最优值时报错，不强行归零。评估 JSON中的旧 `mean_optimality_gap` 是辅助的区域平均指标，主有效区域累计gap以比较脚本输出为准。

## 时间与最优性的边界

**飞行时间**：无人机执行含返航路线的时间，不含巡检悬停。**运行时间**：电脑计算路线的时间，不是飞行耗时，模型不包含训练时间。

网络计时范围：一次原有贪心前向、逐点解码和路线取回CPU；CUDA前后同步。不计模型加载、特征构造、随机基线、绘图或文件输出，不增加预热/重复计时；首次推理初始化可能较慢，设备如实记录。

分支定界计时范围：初始可行路线构造加搜索/证明全过程，包含过程中的正常日志和检查点开销；不计源文件读取、绘图、最终结果导出及程序停止期间。两者范围和设备可能不同，不作为同硬件受控速度实验。

保持原有非对称矩阵和续航检查，不对称化、不取整。自编边包含/排除分支、指派松弛下界和子环剪枝；SciPy仅求指派子问题，不调用Gurobi。最优性是记录的浮点容差内证明，不是实数成本的符号证明。最坏情况仍是指数级，不承诺可预估时间内完成。

## 测试与兼容

```bash
python -m unittest discover -s "stage2/Branch and Bound/tests" -v
python -m unittest discover -s stage2/tests -v
```

覆盖原生Stage1配对、输入冻结、绘图状态、恢复后重画、时间合计、环境不覆盖、全排列最优性、下界和gap分组。训练代码和 Stage1不修改，不自动提交或推送Git。测试/运行产物存专属 `outputs/`，不纳入提交。87测试只用于最终比较，不能反向用于挑模型或调参。
