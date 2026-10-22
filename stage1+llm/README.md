# Stage1+LLM v4：可信工具、多目标部署搜索与实时日志

仅运行本目录的独立部署搜索，不调用原Stage1、不修改Stage2、不训练神经网络。框架仍为 **LLM生成完整部署程序 → 独立审核 → Pareto档案 → 按反馈进化**，不是MCTS。CLI默认读取 `config/search-v4.json`。v3结果可查看和选择，但不能直接恢复为v4搜索；旧配置保留，修改实现后也必须新建实验。

## 1. 安装、密钥与离线测试

下面命令均从仓库根目录运行，Python 3.10+、CPU即可。

```bash
python -m pip install -r "stage1+llm/requirements-dev.txt"
python -u -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -v
python -u "stage1+llm/run.py" smoke-test --run-id offline-check-v4
```

测试和smoke-test均使用模拟LLM，不需要密钥、不调用收费API。冒烟测试覆盖工具式生成、失败、修复、折中方案、重复、API异常、恢复、导出，所有产物明确为 `offline_mock`，不能作为正式算法或研究结果；正式export-best拒绝模拟候选。重复运行换run ID，不覆盖旧目录。

覆盖率（主进程和直接单元测试；隔离子进程由单独集成测试验证，不冒称子进程行覆盖率）：

```bash
python -m coverage run --rcfile "stage1+llm/.coveragerc" -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -q
python -m coverage report --rcfile "stage1+llm/.coveragerc"
python -m coverage html --rcfile "stage1+llm/.coveragerc"
```

正式搜索才需要服务器本地的 `stage1+llm/config/deepseek.local.json`。已有文件不要覆盖；没有时复制example并编辑api_key、base_url以及账号实际可用的model ID。私密文件不通过Git同步，不要把密钥写进命令行、截图或公开示例。实现验收不调用真实API，也不验证模型ID可用性。

## 2. 数据隔离与统计口径

默认真实102＋8个训练KDE＋8个验证KDE，种子1234/1001234，冻结到 `outputs/prepared-v4/manifest.json`。所有候选使用相同实例；87仅用于防混入检查，不向LLM提供其坐标或成绩。验证拓扑参与反馈，因此不是独立最终泛化测试。想评价泛化应在算法选定后另用未参与反馈的留出实例，不据87成绩反复挑选。

```bash
python -u "stage1+llm/run.py" prepare-data --search-config "stage1+llm/config/search-v4.json"
```

硬约束不变：半径7km、空速23m/s、平均风6m/s、保守审核速度17m/s、续航2460s、悬停10s、系数1.14。审核闭环为确定性欧氏最近邻，包含机巢成员的悬停，风机须唯一覆盖、机巢归属自身。有向代理采用当前Stage2口径，0北90东、笛卡尔风角90−罗盘角，速度为23＋6×cos(方向差)；不对称化。固定阵风12m/s只记录，不增加动力学。

必须区分保守续航审核与有向代理优化：不能用较短的有向航时替代保守审核。代理包含最多8个NN起点和20轮有向搬移，返回完整闭环；是启发式，不是最优解。飞行时间不含悬停；保守等效消耗与代理等效消耗分别记录，后者不计机巢任务。

三个主要目标：平均机巢数/风机数、平均代理总航时/风机数、最坏区域非机巢任务数/对应风场规模。完整保存非支配方案，活跃父代池最多12项，不固定八区、不设置任务硬上限、不用隐藏权重替用户选最终方案。

## 3. 候选工具与计算预算

候选仍定义 `solve(turbines, config, seed)` 并返回nest_ids、assignments。LLM决定部署策略；不再重复实现物理公式。受控导入：

执行器兼容 `getattr(config, 'communication_radius_km', 7.0)`，但getattr/hasattr仅能读取配置及风机记录的公开输入字段，不能读取私有字段或反射模块/函数/工具。优先直接使用config属性；未知公开字段可以使用getattr默认值。内置名称清单从执行器同一契约自动附加到LLM提示词；不支持的内置名称或未定义全局名称在执行前明确报错（不保证发现全部运行时错误）。禁止通配符导入，改用显式导入。此次兼容修复改变了实现/提示词指纹，必须新建run ID，不要恢复修复前的实验。

```python
from candidate_tools import region, deployment, split_repair, remember, best, tick, remaining, phase
```

- `region(nest_id,members,proxy=False)`：审核含机巢的完整成员集合，返回可行性、数值违规和安全余量。粗筛须显式传False，实际默认proxy=True，不改变旧工具语义。
- `region(...,proxy=True)`：额外计算统一代理路线及航时。
- `deployment(proposal,proxy=True)`：完整审核，返回三个objectives；proxy=False时不生成目标。
- `distance(a,b)`、`directed_time(a,b)`：km和有向秒。
- `split_repair(assignments)`：可选、显式调用的新增机巢拆分原语，不是优化器，可能增加很多机巢；不要求一次移走一个点立即恢复可行。不自动对所有LLM结果兜底。
- `remember(proposal)`：保存候选自己选择且审核通过的当前解，不隐含质量排序；`best()`返回副本。
- 每个邻域尝试调用 `tick()`；`remaining()`查预算，`phase('阶段')`报告阶段。

距离/有向矩阵每实例预计算，区域按机巢、排序成员与评价模式缓存；环境和工具版本在实例内固定，不跨实例复用。迁移、交换只需评价受影响区域。

v4工程初值：协作式120秒、5000次缓存未命中的区域评价、20000次动作；外部硬超时300秒。预检软预算限制为各自硬超时的80%。这些不是论文参数，也未证明最佳。预算耗尽时仅返回候选显式remember的可行解，没有则失败；硬超时仍失败，父进程最终独立验算。手写计算若不调用工具，不享有缓存或完整工作量统计，仍受外部硬超时约束。

AST限制和独立进程是尽力隔离，不是OS安全沙箱；未知来源代码在生产环境应额外使用容器/低权限账户。候选报告的阶段/计数是诊断，不能替代父进程可行性审核。

预检包括接口和半径/续航反例。任一固定数据集失败可提前停止，明确标注已评估/总实例数，不为未评估项虚构通过。只有全部实例通过才入档。

## 4. 离线旧算法基准（不调用LLM）

归档的C0005/C0008来自用户提供的v3真实搜索，保持原算法语句，不作为新搜索的初始答案。重新测量同一清单，不直接搬旧耗时：

```bash
python -u "stage1+llm/run.py" benchmark --search-config "stage1+llm/config/search-v4.json" --candidate "stage1+llm/baselines/v3/C0005.py" --candidate "stage1+llm/baselines/v3/C0008.py" --run-id baseline-v4
```

输出 `outputs/baseline-v4/benchmark.json` 和实时日志。旧算法特别是C0005可能较慢；没有使用工具的旧代码不受协作式次数预算限制，但有相同硬超时。时间含子进程启动，不是严格硬件测速实验。重复部署但更快的源码保存到效率备选表，不偷偷把耗时变成第四个主要目标。

## 5. 真实搜索：先3轮，再完整运行

从这里开始会调用收费API。省token措施是统一工具、精简反馈和有限历史，不是请求额外模型总结；显示日志本身不额外消耗模型token。生成上限8192不是要求生成这么多，也不是整个搜索总额度。关闭重试不意味着失败请求不计费。

先前台试跑，检查正文、预检、工具调用及阶段日志：

```bash
python -u "stage1+llm/run.py" optimize --search-config "stage1+llm/config/search-v4.json" --run-id trial-v4 --evaluation-budget 3 --no-improvement-limit 3 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code
```

Linux单行后台运行，日志文件按日期命名：

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" optimize --search-config "stage1+llm/config/search-v4.json" --run-id search-v4 --evaluation-budget 40 --no-improvement-limit 10 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/search-v4_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

最多40个逻辑轮次，失败也占轮次；连续10轮前沿无变化提前停止。远端响应不是确定性的，固定本地种子不保证逐字复现。预算不能准确换算运行时或费用。

```bash
tail -f "stage1+llm/outputs/search-v4/search.log"
```

控制台及search.log显示LLM公开正文、源码、请求状态、心跳、预检、逐实例指标、工具阶段/计数、奖励和前沿变化；原始正文保存responses，另有events.jsonl、trajectory.csv、逐轮JSON、pareto.csv和efficiency_alternatives.json。只显示公开回答，不要求或展示隐藏推理。错误、超时也计时，所有结果标记覆盖范围。密钥脱敏不等于磁盘加密。

## 6. 恢复

只能恢复相同v4实现、数据、环境、工具预算、提示词和模型参数；不重复请求已保存的生成结果，也不重复已完成实例。被中断且请求结果未知的轮次不自动重发。未完成的单实例可能从头重跑，其失败/中断尝试耗时单独保留。

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" resume --search-config "stage1+llm/config/search-v4.json" --resume-run-id search-v4 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/resume-v4_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

预算用完可显式加 `--evaluation-budget 60`；因停滞停止，确实需要继续探索时可加 `--reset-stall`。不要同时启动两个进程写同一个run目录。

方向由适用性筛选＋滑动窗口UCB选择，窗口内收益和次数同步更新；初始生成、强制修复单独记录，API错误不计入优化方向统计。完整奖励和目标变动写入日志。保留旧discounted选择器供新实验对照，不宣称新选择器必然更好。

## 7. 选定算法，再最后评估87

```bash
python -u "stage1+llm/run.py" list-candidates --run-id search-v4
python -u "stage1+llm/run.py" export-best --run-id search-v4 --candidate-id C0008 --reason "根据冻结搜索集三目标与运行效率选定"
python -u "stage1+llm/run.py" evaluate --search-config "stage1+llm/config/search-v4.json" --candidate "stage1+llm/outputs/search-v4/selected/C0008/algorithm.py" --input turbine87.tsv --run-id test87-v4-C0008
```

C0008只是示例，必须替换为本次实际选定的候选。只能导出可行前沿成员，不自动根据87选算法。导出核对源代码与可信实现哈希，包含配置、选择记录和工具快照；正式评估拒绝与选择记录不同的物理配置/计算预算。87结果含deployment.png、evaluation.json、summary.json、完整regions.json，可供现有Stage2/分支定界读取；不会自动切换Stage2训练入口。

导出包也可以整体复制到别处，安装NumPy后独立运行：

```bash
python -u "stage1+llm/outputs/search-v4/selected/C0008/run_selected.py" --input turbine87.tsv
```

独立入口在导入快照代码前验证哈希，默认沿用保存的种子/预算，输出JSON；完整绘图和regions导出请使用evaluate。

分区改变后必须重新计算对应分支定界参考，不能沿用旧四区域最优值。当前验证是软件流程验收，不承诺LLM部署质量、泛化能力或优于论文；最终效果须在公平实验中验证。

## 8. 工具契约 v2：此次真实失败的修复

此次只加固接口与诊断，不改变物理约束、三目标、UCB选择或Stage2。区分三类数据：

| 数据 | 形式 | 使用方式 |
|---|---|---|
| 归属映射 | `{"a":"a","b":"a"}` | `split_repair(assignments)`、`make_proposal(assignments)` |
| 完整部署 | `{"nest_ids":["a"],"assignments":{...}}` | `deployment(proposal)`、`remember(proposal)`、最终返回 |
| 评价字典 | `{"feasible":true,"objectives":[...]}` | `checked["feasible"]`、`checked["objectives"][0]` |

`make_proposal()`与`assignments_from_proposal()`返回深拷贝，仅组织和检查结构，不补齐任务、不修复物理违规。`best()`返回保存的完整部署副本或`None`；`remember()`返回布尔值；`remaining()`使用`remaining()["seconds"]`等字符串字段读取。

类型、字段或调用参数错误报`ToolContractError`；不能用`region(...)[0]`、`remaining().get(0)`等元组式读取。正常的目标列表整数索引和`.get("可选字段", default)`仍可用。不将列表、字典、整数ID强制转字符串，也不取列表首元素掩盖错误。

工具先将错误记录到执行器，再抛异常。候选即使捕获异常、返回可行部署或提前保存可行解，本次仍记为`tool_contract_error`，不能进入档案、没有有效目标及正奖励。无接口错误的预算退出仍可按原规则返回显式保存的解。结构正确但缺任务、机巢未归属自身、超半径或超续航仍属于普通不可行，不误报为类型错误。其他内部异常保留真实类别，不一概转成契约错误。

控制台和`search.log`会立即显示工具名、字段、期望类型、实际类型及可取得的候选行号；逐轮JSON保存错误总数和最早5条详情，修复提示仅取最早3条去重诊断与简短正确示例。错误后硬超时仍保留接口诊断与`hard_timeout`记录；被杀死前只能观测到的错误数明确标为下界。`trajectory.csv`另列状态、契约错误数、约束预检和中断检查耗时；这些不混入部署算法的运行时间。

三份用户失败源码以原始字节Base64与SHA256保存在`tests/fixtures/tool_contract_failures.json`，只用于只读回归，不作为初始候选或手工修好的“搜索成功”。字典守卫不承诺追踪任意Python行为，例如显式复制为普通`dict`后的读取；最终结构检查和独立物理验算仍然必需。执行器也不是OS级安全沙箱。

新运行元数据记录`tool_contract_version=2`。实现和提示词指纹已改变，旧实验只能读取，不能直接恢复；冻结数据和配置保持不变。旧导出包使用原工具快照，执行器按工具版本走兼容分支，不改写旧快照。新导出包仍只需原六个运行时模块，包含新增辅助接口、异常和结果对象。

先离线检查（不调用API）：

```bash
python -m pip install -r "stage1+llm/requirements-dev.txt"
python -u -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -v
python -u "stage1+llm/run.py" smoke-test --run-id offline-check-contract-v2
```

同步代码并保留服务器的私密API配置后，新建3轮后台试跑（会产生API费用；不要同时写同一run目录）：

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" optimize --search-config "stage1+llm/config/search-v4.json" --run-id trial-v4-contract --evaluation-budget 3 --no-improvement-limit 3 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/trial-v4-contract_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

```bash
tail -f "stage1+llm/outputs/trial-v4-contract/search.log"
python -u "stage1+llm/run.py" list-candidates --run-id trial-v4-contract
```

若该目录已存在，换一个新run ID及对应日志名，不覆盖旧结果。只有试跑产生可行候选且没有持续接口错误后，再用新run ID开启40轮搜索；接口修复不保证LLM一定生成好算法。

## 9. 历史质量增强：工具 v3（旧搜索使用本节，新实验见第10节）

根据真实`trial-v4-contract`修复：C0002已能运行，但大量实例退化为“一个大区域＋很多单点巢”；C0003在部分拓扑正常优化，另一些在修复后直接返回。此次不改变物理约束、外部三目标、Pareto入档、UCB及Stage2，不把质量诊断当成第四个目标或新的物理硬约束。

### 本次改动

- 旧`split_repair(assignments)`保留原语义；新增显式可选`repartition_repair(assignments)`。只重建违规区域，按原保守审核增量吸收附近任务、逐步打开新巢，保留原本可行区域。它是手写工程启发式，不保证机巢更少或全局最优，不自动保存、不替LLM继续优化；预算耗尽仍遵守原规则。
- 提示词要求初始构造同时考虑半径/续航，机巢数量自适应；修复得到可行解后先remember，再进入候选自己的有界搜索。框架不会替候选补一段隐藏优化，也不因单点机巢合法存在而拒绝入档。新LLM是否实际遵循，需要真实试跑验证。
- 新增`objective_dominates(a,b)`及`prefer_objectives(candidate,incumbent,weights=...,scales=...)`。后者必须显式给三维权重/尺度，先按Pareto支配、折中时按归一化加权和比较，相等保留旧解。实际权重/尺度写入日志和telemetry；只是候选内部偏好，不能改变外部三目标或自动替用户选“最好”。不是论文原参数。
- 工具签名从实际方法生成，字段语义共用一处说明；距离工具明确只收风机ID字符串。修复反馈按错误工具补短说明和出错行附近源码，保留完整父代代码但不重复整段堆栈；质量反馈只保留紧凑指标，不把长区域列表反复送给LLM。

### 看什么日志和结果

控制台及`search.log`增加`repair_start`、`repair_end`、`objective_preference`、`quality_diagnostic`，保留公开LLM流式输出、完整源码、心跳、预检和逐实例验算。

每实例`quality`记录任务分布、零非机巢任务区域比例、任务数变异系数、修复工具调用、最后一次成功修复后的外部评价调用数及不同可行归属的保存次数。`remember()`内部审核不冒充修复后的外部搜索评价。检查过不同解或保存过不同解，都不等于目标一定改善；单改phase名字也不算证明。

- `zero_task_regions_present`：有仅含机巢自身的区域，可能合理，不代表漏覆盖或不可行。
- `repair_return_without_external_checks`：正常返回，但最后成功修复后未观察到候选调用region/deployment；提示可能直接交付修复解，不作为可行性否决。
- `repair_exit_without_external_checks`：预算等退出时未观察到后续评价，不能直接指责为提前放弃优化。
- 旧工具快照没有新计数时，标记`repair_observability=unavailable`，不估算历史搜索次数。

新增`quality_report.json`、`quality_summary.csv`；训练/验证分别及全部实例汇总，标出已覆盖/应覆盖/可行数量，失败或未覆盖不虚构通过。`evaluations/*.json`保留逐实例完整质量记录，`pareto.csv`和`list-candidates`补充零任务区域及修复后无后续评价数量。反馈优先展示不可行及质量退化实例，合法的折中方案仍完整保留。

工具版本升为3，搜索CLI/配置入口保持不变；旧运行可读、旧导出仍使用旧快照，但指纹改变后不能恢复旧搜索。使用新run ID，冻结数据仍沿用`prepared-v4`，不得清空或覆盖以掩盖差异。

### 先离线测试与对照（无API费用）

```bash
python -m pip install -r "stage1+llm/requirements-dev.txt"
python -u -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -v
python -u "stage1+llm/run.py" smoke-test --run-id offline-quality-v3
```

以下命令在同一冻结102及16个KDE实例上比较旧/新修复工具和原始C0002/C0003，不运行87，不调用LLM，结果不能冒充新LLM算法成绩：

```bash
python -u "stage1+llm/run.py" benchmark --search-config "stage1+llm/config/search-v4.json" --candidate "stage1+llm/baselines/quality/split_only.py" --candidate "stage1+llm/baselines/quality/repartition_only.py" --candidate "stage1+llm/baselines/trial-contract/C0002.py" --candidate "stage1+llm/baselines/trial-contract/C0003.py" --run-id quality-baseline-v3
```

输出`outputs/quality-baseline-v3/benchmark.json`。四个程序共享该文件记录的数据指纹；若与历史服务器试跑指纹不同，不宣称是同一历史数据的直接数值复现。旧/新工具都用同一硬约束验算；不得仅凭更低飞行时间忽略机巢数增加。

### 服务器后台试跑6轮（调用收费API）

代码同步并保留本机私密配置后，从仓库根目录执行：

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" optimize --search-config "stage1+llm/config/search-v4.json" --run-id trial-v4-quality-v3 --evaluation-budget 6 --no-improvement-limit 6 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/trial-v4-quality-v3_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

```bash
tail -f "stage1+llm/outputs/trial-v4-quality-v3/search.log"
python -u "stage1+llm/run.py" list-candidates --run-id trial-v4-quality-v3
```

6是逻辑轮数，失败仍占轮次；初始生成和强制修复不作为普通UCB优化证据。应检查是否出现非修复的父代优化轮，以及全部实例可行、单点巢比例、最差实例和三目标折中，不能仅看“前沿有候选”。试跑质量稳定后才新建40轮正式实验，勿按87成绩反复挑选。

同一新版本正常中断可恢复，不要同时写同一目录：

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" resume --search-config "stage1+llm/config/search-v4.json" --resume-run-id trial-v4-quality-v3 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/resume-quality-v3_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

## 10. 当前版本：含风况Stage2反馈驱动的部署进化（v5）

本节替代新版的三目标Pareto选优。旧实验仍用原配置、结果和工具快照，不会自动变成v5；修改了实现指纹的旧搜索不能直接恢复。所有命令从 `wind-turbine` 仓库根目录执行，只在 `stage1+llm/` 保存新结果。

### 算法逻辑与约束

LLM生成完整 `solve(turbines, config, seed)` → 得到机巢和归属 → 检查覆盖、自身归属、服务半径 → 冻结Stage2单次贪心规划 → 按实际路线计算含风有向飞行时间、返航及续航 → 反馈给LLM迭代。

- 正式权重：`stage2/outputs/finetune-edge-v2/control/best_greedy.pt`。只加载一次，不训练、不采样、不增加后处理，也不回退为最近邻。CPU必须显式指定，CUDA不可用不会静默切换。
- 默认风速6 m/s、风向90°、阵风特征12 m/s、空速23 m/s、半径7 km、续航2460 s、悬停10 s、等效系数1.14。直接复用Stage2风向转换、特征、有向时间矩阵及路线验算，不另加禁飞方向或阵风动力学。阵风是特征，不是本次新增的随机过程。
- 删除v5所有工具/修复/最终审核中的最近邻巡检路线与启发式代理。几何距离可辅助分区，不等于使用最近邻路线。续航只按网络实际闭合路线加非机巢任务悬停消耗计算；零外出任务区域飞行及任务悬停均为0。
- 所有固定实例可行后依次比较：平均机巢数 → 平均全场总飞行秒 → 全部实例中最长区域飞行秒，同分保留较早候选。每实例平均区域飞行秒=`总飞行秒/机巢数`，仅展示，不重复计分。机巢数少绝对优先于飞行时间改善，不是加权或Pareto。
- 网络未取得可行路线不等于部署已证明无解；结果只能称“固定网络与预算内找到的最佳可执行算法”，没有全局最优证明。
- 使用102+8训练KDE+8选择验证KDE，沿用冻结数据和种子。选择验证数据会参与反馈，不是独立最终测试；87只能在算法选定后评价。

工具版本4无 `proxy` 参数，也不再暴露 `objective_dominates` / `prefer_objectives`。`region()`始终返回实际网络路线评价；`deployment()`返回 `metrics` 及可行时的 `objectives=[机巢数,总飞行秒,最长区域飞行秒]`；`better_metrics(a,b)`按优先级比较。`remember()`只保存显式选定且可行的方案，不自动选优。工具契约错误即使被候选捕获也使该次执行失败。

保留40轮预算、3个独立初始候选、连续10轮最佳无改善停止。父代池最多6个，70%选最佳、30%选其余高排名方案，失败修复和UCB方向选择保留。软预算120秒（含推理等待）、硬超时300秒、未缓存区域评价5000次、动作尝试20000次。失败仍占逻辑轮数，SDK请求重试与逻辑轮次分开；模型/显存/服务故障停止并保留pending源码，不自动增加API请求。

### 环境与离线检查（不收费）

先激活服务器已有的CUDA版PyTorch环境，不通过下面依赖安装强制替换PyTorch。配置私密DeepSeek文件，勿提交密钥。

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -m pip install -r "stage1+llm/requirements-dev.txt"
python -u -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -v
python -u "stage1+llm/run.py" smoke-test --run-id offline-joint-v5
```

测试中的微型随机权重和模拟规划器不是正式模型，也不能证明部署性能。冒烟流程明确标记 `offline_mock`，覆盖错误、修复、去重、六轮搜索、恢复及导出，不调用API。已有同名目录时更换run ID，不覆盖旧结果。

正式模型在启动搜索时先检查文件、架构、设备和完整环境，成功后才允许请求LLM。checkpoint环境明确不一致会报错；缺少环境记录会警告并使用显式配置，不会静默改变风向。正式模型仅存在服务器时，本地测试不冒充服务器验收。

服务器可先进行不收费的权重与接口检查：对输入的前4个节点，核对直接Stage2贪心与常驻工作进程的路线、时间完全一致。这不是完整风场成绩或性能证明：

```bash
python -u "stage1+llm/run.py" check-model --stage2-checkpoint "stage2/outputs/finetune-edge-v2/control/best_greedy.pt" --stage2-device cuda
```

覆盖率（进程外部工具调用不自动计入父进程覆盖率）：

```bash
python -m coverage run --branch --source="stage1+llm/stage1_llm" -m unittest discover -s "stage1+llm/tests" -p "test_*.py"
python -m coverage report -m
```

### 六轮后台试跑（会调用收费API）

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" optimize --search-config "stage1+llm/config/search-v5.json" --stage2-checkpoint "stage2/outputs/finetune-edge-v2/control/best_greedy.pt" --stage2-device cuda --run-id trial-joint-v5 --evaluation-budget 6 --no-improvement-limit 6 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/trial-joint-v5_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

```bash
tail -f "stage1+llm/outputs/trial-joint-v5/search.log"
python -u "stage1+llm/run.py" list-candidates --run-id trial-joint-v5
```

控制台展示公开LLM流式回答、完整代码、10秒心跳、预检、逐区域网络时间/能耗/余量、逐实例总/平均/最大时间及父子代比较。缓存命中使用汇总计数，原始单次推理耗时不当作重复推理耗时累加。

结果包括 `search_state.json`、`summary.json`、`evaluations/*.json`、`events.jsonl`、`trajectory.csv`、`ranking.json`、`ranking.csv` 和 `best.json`。不再输出新版Pareto表。每个可行候选的逐实例完整分区与路线保存在逐轮JSON；选定后的评价另输出冻结清单、路线图和模型指标。

### 正式40轮及恢复

试跑没有持续接口错误且正式模型评价稳定后，再启用新run ID：

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" optimize --search-config "stage1+llm/config/search-v5.json" --stage2-checkpoint "stage2/outputs/finetune-edge-v2/control/best_greedy.pt" --stage2-device cuda --run-id search-joint-v5 --thinking disabled --max-tokens 8192 --api-retries 0 --stream --show-code > "stage1+llm/outputs/console/search-joint-v5_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" resume --resume-run-id search-joint-v5 --stage2-checkpoint "stage2/outputs/finetune-edge-v2/control/best_greedy.pt" --stage2-device cuda > "stage1+llm/outputs/console/resume-joint-v5_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

正常恢复不重复已完成实例，也不重新付费生成已有源码。未知结果的中断API请求消耗该轮、不自动重放。恢复核对数据、实现、提示词、checkpoint哈希、设备和完整环境；改变口径需新建实验。同一run ID禁止并发写入。

每个搜索使用排他锁 `outputs/.运行编号.lock`。正常退出自动释放；若进程被强制终止留下锁，确认该编号没有运行进程后才能手动删除锁再恢复。程序不会自动清理无法确认归属的旧锁。候选超时后会先终止候选并回收尚在途的网络响应；监督器可能继续等待至该请求原有的300秒服务故障时限，因此整次监督器墙钟耗时不等于严格的候选执行时限。只有工作进程自身超过服务时限或异常才停止整个搜索。

### 导出、最后评价87与分支定界

```bash
python -u "stage1+llm/run.py" export-best --run-id search-joint-v5
```

默认导出当前排名第一者；也可显式 `--candidate-id C0007` 导出已完整评价的可行候选。把下面示例编号换成导出命令实际返回的目录编号，不依据87成绩反复选算法：

```bash
python -u "stage1+llm/run.py" evaluate --search-config "stage1+llm/config/search-v5.json" --candidate "stage1+llm/outputs/search-joint-v5/selected/C0007/algorithm.py" --input turbine87.tsv --stage2-checkpoint "stage2/outputs/finetune-edge-v2/control/best_greedy.pt" --stage2-device cuda --run-id test87-joint-v5
```

输出 `summary.json`、`regions.json`、`model_metrics.json`、`evaluation.json` 和逐区域网络路线图。所有风机、机巢、坐标及风况冻结在清单里。新导出包记录可信工具快照、模型哈希和依赖，运行需要匹配的Stage2仓库及单独提供的权重，不再宣称无需模型即可独立执行。

新部署改变了分区，必须针对新的 `regions.json` 重新求分支定界参考，不能沿用旧四区域最优值。最优性gap只在相同区域与环境下比较；分支定界不参与LLM搜索反馈。
