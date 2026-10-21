# Stage1+LLM v3：多目标部署算法搜索

只在本目录运行与保存结果。不修改、不调用原 Stage1 部署算法，不使用 Stage2 网络评分，也不训练神经网络。LLM生成完整 `solve(turbines, config, seed)` 程序，可信父进程独立验算；最终由你手动选择算法。

## 1. 环境和密钥

在仓库根目录使用 Python 3.10+（CPU即可）：

```bash
python -m pip install -r "stage1+llm/requirements-dev.txt"
```

正式搜索才需要在本机创建私密配置：

```bash
cp "stage1+llm/config/deepseek.example.json" "stage1+llm/config/deepseek.local.json"
nano "stage1+llm/config/deepseek.local.json"
chmod 600 "stage1+llm/config/deepseek.local.json"
```

Windows可以复制示例后用文本编辑器编辑。填写真实 `api_key` 及**账号当前可用的模型ID**；示例中的模型名称不代表已验证可用。本实现未调用真实API验收。密钥仅从这个JSON读取，不会从Git下载，不要放进命令行、截图或聊天。不要用公开示例覆盖已有私密配置而丢失密钥。

省token缺省设置：`thinking: "disabled"`、`reasoning_effort: null`、`max_tokens: 8192`、`retries: 1`、`stream: true`。显式关闭高耗token思考模式，保留完整代码，四项公开说明各不超过80字；父代反馈去掉完整分区及重复信息，仍保留失败实例的数值诊断。**这些措施不保证固定费用或算法质量**。原私密文件显式设置的额度仍生效，可用下方命令覆盖；无需修改密钥。

## 2. 先跑离线测试：不需要密钥，不收费

```bash
python -u -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -v
python -u "stage1+llm/run.py" smoke-test --run-id offline-check-v3
```

冒烟测试模拟5轮：失败部署、修复、另一种折中部署、重复方案、生成失败；还检查恢复与导出。所有结果标为 `offline_mock`，不是正式算法或研究结果；正式 `export-best` 会拒绝模拟候选。重复运行请换run ID，不能覆盖旧结果。

覆盖率（输出全部位于本目录）：

```bash
python -m coverage run --rcfile "stage1+llm/.coveragerc" -m unittest discover -s "stage1+llm/tests" -p "test_*.py" -q
python -m coverage report --rcfile "stage1+llm/.coveragerc"
python -m coverage html --rcfile "stage1+llm/.coveragerc"
```

核心逻辑最低80%；图形布局及已退休的种子模板不计入该门槛。Windows测试依赖正常可写的系统临时目录。候选解释器显式UTF-8，避免中文路径下Conda `.pth` 的GBK解码问题。

## 3. 准备冻结数据

```bash
python -u "stage1+llm/run.py" prepare-data
```

默认配置在 `config/search.json`：真实102＋8份训练KDE＋8份验证KDE，共17个实例；随机拓扑各102点，种子1234和1001234。保存在 `outputs/prepared-v3/manifest.json`。

同一搜索的所有候选都使用同一份数据，不因候选失败重新抽样。清单记录坐标、逻辑来源、源文件哈希、种子与内容指纹。重复准备只校验，不覆盖。改变数据/种子时需指定新 `data_dir`。默认不再读取189或500参与搜索。

搜索输入会与87坐标核对，拒绝把87或包含它的189当搜索数据。保护检查不把87坐标或成绩发送给LLM。验证集参与选优，不能称为最终盲测集。

## 4. 先真实1轮连通性试跑，再运行完整搜索

以下命令会调用收费API，由你决定何时运行：

```bash
python -u "stage1+llm/run.py" optimize --run-id trial-v3-stream --evaluation-budget 1 --thinking disabled --max-tokens 8192 --api-retries 1 --stream --show-code
```

完整前台运行，逐轮直接显示在控制台：

```bash
python -u "stage1+llm/run.py" optimize --run-id search-v3-stream --thinking disabled --max-tokens 8192 --api-retries 1 --stream --show-code
```

1轮仅用于验证“请求→正文→代码检查→17份数据评估”能否完成，不代表算法已经搜索充分。默认完整搜索3个独立初始候选、最多40轮、前沿连续10轮无改善停止。生成失败占1轮；每轮SDK重试可能增加实际HTTP请求。日志记录实际尝试数及最终响应可获得的token用量，**不是账单，也不保证覆盖所有重试的计费用量**。`--api-retries 0` 可禁用传输重试；不会因解析失败或断流自动重新付费生成。

Linux后台单行命令，外层日志以日期命名：

```bash
mkdir -p "stage1+llm/outputs/console" && nohup python -u "stage1+llm/run.py" optimize --run-id search-v3-stream --thinking disabled --max-tokens 8192 --api-retries 1 --stream --show-code > "stage1+llm/outputs/console/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

实时看每轮过程：

```bash
tail -F "stage1+llm/outputs/search-v3-stream/search.log"
```

日志为UTC时间，示意（非实验成绩）：

```text
[iteration_start] round=... parent=C.... direction=... reason=...
[llm_start] C.... LLM request start
[llm_request_config] C.... thinking=disabled effort=None max_tokens=8192 stream=True retries=1 prompt_chars=...
[api_request] C.... Provider HTTP request attempt=...
[heartbeat] C.... LLM request active; elapsed=...s
[llm_stream_start] C.... Provider stream connected; receiving response (not yet validated)
[llm_text] C.... {"algorithm_idea":"...", ...
[llm_text] C.... ...增量返回的公开JSON正文及源码...
[llm_response] C.... finish_reason=... response_chars=... usage=... public_response=responses/C....json
[llm_explanation] C.... algorithm_idea: ...
[llm_code] C.... returned source BEGIN (not executed/validated yet)
[llm_code] 0001 | def solve(turbines, config, seed):
[dataset_start] C.... dataset=... nodes=102 start
[dataset_end] ... feasible=... nests=... proxy_flight=... max_tasks=... algorithm=... evaluator=...
[iteration_end] ... reward=... decision=... objectives=... delta=... front=... stalled=...
[search_end] ... stop=budget_exhausted/front_stalled
```

等待API及运行算法/评估器时每10秒心跳；`--heartbeat-seconds` 可覆盖。方向分解保存在结构化事件和逐轮记录。日志、控制台和事件都做密钥脱敏；不输出隐藏推理过程。

### LLM返回内容可见性与JSON错误排查

默认 `show_code: true` 且 `stream: true`：公开正文边生成边通过 `[llm_text]` 显示，完成后再打印四项说明（算法思路、修改假设、预期改善、修改摘要）及完整带行号源码。代码展示发生在约束评估前，**看到代码不代表代码已执行成功或已证明可行**。可加 `--no-show-code` 同时关闭原始正文和源码展示；四项说明、响应诊断和源码文件仍保留。`--no-stream` 则等完整响应后显示公开正文。

为便于阅读和跨分块脱敏，公开正文合并成小段打印（约256字符，或有新分块到达且距上次显示已过1秒；密钥前缀会延迟到可安全脱敏时显示），并非每token打印一行。原始JSON中的代码换行可能显示为 `\n`，完成后的带行号源码会正常换行。首字到达之前只有心跳，心跳不是供应商的百分比进度，也无法展示服务端尚未返回的文字。只处理公开的 `delta.content`/`message.content`，不读取或输出内部推理字段。打印内容不增加API调用或模型token。

每次收到响应，会在解析JSON前保存 `responses/C0001.json` 等诊断文件，包含脱敏后的公开正文、结束原因、响应ID、正文字符数和可获得的token用量。流式中断时也保存已接收的公开正文，并标记 `stream_complete: false`；截断、断流、未收到结束标志或JSON错误均不得进入执行。过程中已显示片段保存在 `search.log` 与 `events.jsonl`；断电/强杀只能保留此前已落盘的片段。不保存整个供应商响应对象或内部推理字段。

- `finish_reason=length`：响应被截断，拒绝执行；需要结合 `max_tokens`、上下文和token明细判断，不自动增大预算或重发。
- 正文为空：明确报告 `empty final content`，而不是笼统JSON异常。
- 非法JSON：显示行、列及字符数，在控制台展示公开正文前12000字符，完整正文仍在诊断文件。
- 无choices、过滤或其他中断：分别报告，不冒充可部署候选。

这次仅修改调用预算、公开输出、提示词紧凑程度和完整性检查，不改变部署硬约束、三目标或进化评分。旧版本未保存的失败响应无法事后恢复。此更新改变了实现和提示词哈希，旧搜索不能混用新实现续跑；先保留旧任务产物，新版本使用新run ID，例如 `trial-v3-stream`。无需覆盖旧目录。

## 5. 看方案、手动选定、最后评估87

```bash
python -u "stage1+llm/run.py" list-candidates --run-id search-v3-stream
```

控制台显示候选编号、跨实例平均机巢数、平均总飞行时间（启发式）、最大区域任务数和三目标。逐数据集原始指标在 `evaluations/*.json`，不要把跨实例平均数当成87成绩。

例如看完表后选C0007（**请替换成真实存在且仍在前沿的编号**）：

```bash
python -u "stage1+llm/run.py" export-best --run-id search-v3-stream --candidate-id C0007 --reason "根据机巢成本和最大区域规模选择"
```

输出 `outputs/search-v3-stream/selected/C0007/algorithm.py`、选择记录及已评估的指标。没有默认的 `best_algorithm.py`，也没有隐含加权选模。

固定选择后评估87，不回写搜索奖励：

```bash
python -u "stage1+llm/run.py" evaluate --candidate "stage1+llm/outputs/search-v3-stream/selected/C0007/algorithm.py" --input turbine87.tsv --run-id test87-C0007
```

可行时输出：

- `evaluation.json`：约束、代理飞行时间、部署算法耗时及评估器耗时。
- `summary.json`：兼容原生Stage1摘要的固定部署，含保守审核路线和指标。
- `regions.json`：包含完整坐标、机巢、环境和实例指纹的固定区域清单。
- `deployment.png`：机巢部署图。

不可行返回退出码2，不导出虚假的可行摘要或区域清单；不覆盖同名评估。

可将这份新 `regions.json` 交给现有Stage2评估和分支定界入口，但**Stage2默认训练数据生成器仍使用原Stage1**，本次没有自动替换它。机巢/分区变化后，必须为新清单重新计算分支定界参考；不能沿用旧四区域最优值。判断网络能力时，各网络必须使用同一清单及相同解码预算。

## 6. 恢复规则

```bash
python -u "stage1+llm/run.py" resume --resume-run-id search-v3-stream --thinking disabled --max-tokens 8192 --api-retries 1 --stream --show-code
```

增加预算（只能增大，其他实验身份保持不变）：

```bash
python -u "stage1+llm/run.py" resume --resume-run-id search-v3-stream --evaluation-budget 60 --reset-stall --thinking disabled --max-tokens 8192 --api-retries 1 --stream --show-code
```

若3轮试跑覆盖了 `--no-improvement-limit 3`，恢复也须带同值；正式默认是10。恢复核对数据哈希、物理约束、目标、模型配置、提示词及实现哈希，拒绝不同实验混用。数据内容相同而目录搬迁不影响逻辑身份；源文件的字节哈希变化仍会被拒绝。

已完成的轮次及已落盘的数据集评估不重复。已保存的LLM返回源码可恢复，不重新请求API。若中断发生在请求期间且尚无返回结果持久化，该轮记为结果未知并消耗预算，不自动重复同一请求；已产生的供应商费用无法撤销。恢复时间累计实际运行时间，不包括进程停止期间。

v2与更早的实验目录保留不动，但不能按新目标续跑；请新建run ID。`prepare-data` 原子发布清单，损坏清单会报错，不静默覆盖或换数据。不要并发启动同一run ID。

## 7. 算法与指标说明

硬约束：全覆盖、唯一归属、机巢来自输入且归属自身、通信/有效半径、保守最近邻闭环等效续航。保守速度是23−6=17m/s；悬停等效时间10×1.14=11.4秒，审核计入机巢风机。

有向代理与当前Stage2一致：罗盘0北90东转笛卡尔角，边时间=距离米÷(23+6×方向余弦)，两个方向不对称。候选路线包含保守审核路线和最多8个确定性起点，最多20轮有向单点搬移。代理悬停只计非机巢任务，飞行时间不含悬停。12m/s阵风仅为固定环境特征，不是随机阵风仿真。代理不是最优解，也不保证等于将来训练网络的成绩。

三项均最小化，全部固定实例等权：

1. `mean(机巢数/风机数)`；
2. `mean(各区域代理飞行时间合计/风机数)`；
3. `max(单区域非机巢任务数/对应风场风机数)`。

无其他方案三项都不差且至少一项更好的候选，进入Pareto前沿。保留全部非支配候选；活跃父代池最多12，拥挤距离保留多样性。每4个进化轮次有修复池则安排一次修复，无可行父代时优先修复，不强行凑8区。所有实例通过才可获正奖励，重复方案不获正奖励。

本框架可能保留“很多机巢、很小区域”的极端方案，这是成本/任务规模的折中，不代表整体算法更好。**巢所在风机目前视为已在本地处理、不需要Stage2飞行访问**；增加机巢还会减少飞行任务数，比较时必须连同机巢成本一起报告。安全余量和规模分布始终单独保存。

主要产物：`search.log`、`events.jsonl`、`search_state.json`、`config_snapshot.json`、`trajectory.csv`、`pareto.json/csv`、逐轮源码与评估。模拟和正式来源分开；没有方案时明确报告 `no_deployable_llm_candidate`，不存在回退人工算法冒充LLM的逻辑，也不声称全局最优。

候选解释器具有AST导入/属性限制及超时，但**不是OS级安全沙箱**，没有内存/文件系统隔离保证。只运行可信来源的生成程序，服务器建议使用非特权用户或容器；不要将来源不可信的人类程序当安全测试输入。
