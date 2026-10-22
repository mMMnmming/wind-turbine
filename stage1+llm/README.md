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

- `region(nest_id,members,proxy=False)`：审核含机巢的完整成员集合，返回可行性、数值违规和安全余量。
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
