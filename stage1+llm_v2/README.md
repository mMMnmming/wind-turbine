# Stage1 + LLM v2：从零生成、固定边界的部署算法进化

这是独立的新目录，不导入 `stage1+llm/`，不依赖 C0003，不修改原 Stage1、Stage2 或正式权重。

LLM 生成的是完整 Python 搜索策略模块，而不是某个实例的固定答案。框架运行策略、评价候选、接受严格改善。内层运行不再调用 LLM。

## 工作流程

1. 加载正式 Stage2 权重，检查 CUDA、架构及权重记录的风况环境；此时没有 API 请求。
2. 第一代依据任务、固定接口和约束从零生成代码，不提供旧候选。
3. 每代、每个实例都重新构造并验证“一台风机一个巢”的初解。
4. 生成代码提出完整归属；框架检查覆盖、合法巢、自归属、服务半径，再调用冻结 Stage2 单次 greedy 规划。
5. 按实际含风闭环路线、返航、非巢任务悬停等效消耗检查每区域续航。
6. 仅严格改善时保存并更新当前部署。相同巢数按总飞行时间比较，再比较最长区域时间。
7. 独立冷缓存审核最终最好部署，保存结果，再把精简真实反馈交给 LLM 生成下一代代码。

候选执行在受限独立子进程中，不能导入评价器或 Stage2，不能访问文件、网络、进程和私有属性；不是操作系统级安全沙箱。硬约束、接受规则、预算和最好解由框架掌握。

## 代码接口

完整源码必须提供：

```python
def initialize(problem, initial_solution, seed):
    return {"cursor": 0}  # 示例状态，不是部署算法

def propose(problem, current_solution, state, feedback):
    return {
        "assignments": None,  # 实际策略须提交完整 dict[str,str]
        "state": state,
        "phase": "结束",      # 自报阶段，不作为实际执行证据
        "done": True,
    }
```

`current_solution` 是已接受的最好解副本，包含完整 `assignments`、`nest_ids`、`metrics`、`objectives`。改输入副本不会影响框架；必须返回完整候选。

`problem` 含风机坐标、部署配置、固定风况，以及由 Stage2 相同风模型计算的有向航段时间矩阵及 ID 顺序。矩阵可用于构造/排序，不能替代网络实际路线审核。

`feedback` 包括上次是否可行/被接受、目标、主要拒绝原因、最长/最紧区域、剩余时间/动作/推理/停滞预算。策略用 JSON 可序列化 `state` 推进游标，不能把 `Random`、集合或迭代器放进状态。

可选 `candidate_ops` 的 `merge/relocate/remove/reassign` 是纯结构变换，不选目标、不修复、不搜索。也允许自行组织完整归属。具体契约见 `prompts/system.md`。

## 选优与效率

完整实例集全部通过且代码无异常，候选才可进入算法档案。排序固定为：

1. 平均巢数最小；
2. 平均全场总飞行时间最小；
3. 全部实例最长区域飞行时间最小。

保留逐实例巢数。总飞行时间/巢数仅用于展示。运行时间单独报告，不增加加权目标。

每实例记录策略计算时间、框架评价时间、监督/序列化等开销、总搜索时间、独立最终审核时间。LLM 生成时间独立记录。每代各实例均采用冷缓存起点，实例内允许缓存，最后另作冷缓存审核；推理计数区分搜索与最终审核。

`best_at_seconds`、首次改善时间、不同候选、可行候选、严格改善次数及诊断中的改善曲线用于判断效率。即使完整初解可行，也会标记 `only_initial_solution`；代码异常后保住方案不代表代码成功。

## 日志三分

```text
outputs/<run-id>/
  progress.log                    # 人看：阶段、心跳、关键改善和成绩
  summary.json                    # 人看/交给助手：整体状态与最好结果
  trajectory.csv                  # 人看/交给助手：逐代成绩和耗时
  instance_metrics.json           # 人看/交给助手：紧凑逐实例指标
  candidates/C0001.py              # 每代完整源码
  llm_feedback/C0001.json          # LLM看：实际发送的结构化反馈
  diagnostics/
    events.jsonl                  # 助手按需看：完整事件流
    requests/                     # 实际提示词、反馈、生成配置
    responses/                    # 原始公开响应、用量、完成状态
    evaluations/                  # 逐实例完整结果、路线审核、改善过程
    tool_calls/                   # 策略输入输出、Stage2调用记录
    objects/                      # 按内容指纹去重的完整数据对象
    errors/                       # 错误、行号、堆栈与上下文
    runtime/                      # 配置、模型、数据/代码指纹、恢复状态
```

终端只展示简洁日志，不打印源码、API响应或逐航段数据。默认每15秒输出当前阶段与已知最好成绩；大规模初期减巢合并进度，首次减巢和10巢以内的减巢立即显示。同巢数的频繁改善写入完整记录并在心跳中汇总。长时间生成、候选计算、Stage2评价和最终审核均有心跳。

完整记录不等于全部送给LLM。LLM只接收选定父代源码、紧凑逐实例指标、主要瓶颈、最近错误和真实改善经验。完整诊断可能较大，保留在服务器，日常只传顶部三个结果文件。响应仅记录公开内容，不额外请求或导出隐藏思维链。密钥不落盘。

## 停止与代码进化

- `strategy_declared_neighborhood_exhausted`：策略声明当前邻域搜索结束，不是全局最优证明。
- `stagnation_limit`：连续一定动作没有严格改善，包括重复或空提交。
- `time_budget_exhausted` / `proposal_budget_exhausted` / `inference_budget_exhausted`：达到对应预算。
- `candidate_error` / `candidate_timeout`：代码异常/回调未在时间内返回，单独标记失败。

初始默认每实例搜索120秒、20000次提交、5000次Stage2推理、1000次连续无改善，先达到者停止。初解验证和最终独立审核在搜索预算之外，单独计时。改进是对已接受部署的严格改善，不允许劣解成为当前解。

外层默认6次请求，SDK重试固定为0，失败消耗本轮，没有额外修复请求。没有父代时从零生成；出现错误，下一轮携带出错源码和具体错误修复；已有正常候选时进行结构变异、精英改进、参数改进，有两个可用父代后可融合。保留多个成绩较好的父代，不保证每个子代更优。

## 服务器运行

在仓库根目录、现有 Stage2 CUDA 环境中执行。以下命令由操作者运行；本地开发不会调用收费API。

```bash
python -m pip install -r "stage1+llm_v2/requirements.txt"

python -u "stage1+llm_v2/run.py" prepare-data

python -u "stage1+llm_v2/run.py" check-model \
  --stage2-device cuda \
  --run-id "v2-check-$(date +%Y%m%d_%H%M%S)"
```

`check-model` 验证正式权重、设备、环境，以及前4个节点的直接/桥接贪心路线、含风航时、能耗审核一致；不调用LLM，也不代表部署性能。

私密配置可继续通过参数读取旧目录的现有文件，无需复制，也不会修改它。v2在内存中固定使用 `retries=0`，不修改私密配置；也可根据 `config/deepseek.example.json` 创建被本目录 `.gitignore` 排除的 `config/deepseek.local.json`。不要将密钥写进命令行或提交Git。

模型验收通过后，由操作者启动6轮：

```bash
run_id="v2-$(date +%Y%m%d_%H%M%S)"
mkdir -p "stage1+llm_v2/outputs/console"
nohup python -u "stage1+llm_v2/run.py" optimize \
  --run-id "$run_id" \
  --deepseek-config "stage1+llm/config/deepseek.local.json" \
  --stage2-device cuda \
  --evaluation-budget 6 --thinking disabled --max-tokens 8192 --stream \
  > "stage1+llm_v2/outputs/console/${run_id}.log" 2>&1 < /dev/null &
echo "$run_id"
```

观察必要进度：

```bash
tail -f "stage1+llm_v2/outputs/$run_id/progress.log"
```

中断后使用相同运行ID、配置和生成参数：

```bash
python -u "stage1+llm_v2/run.py" resume \
  --run-id "$run_id" \
  --deepseek-config "stage1+llm/config/deepseek.local.json" \
  --stage2-device cuda \
  --evaluation-budget 6 --thinking disabled --max-tokens 8192 --stream
```

已保存源码不重新付费生成，已完成实例不重复运行。中断的内层实例从单巢初解重新执行。请求结果未知且没有完整响应时，该轮标记未知并消耗预算，不重发同一请求；恢复后仍可继续剩余轮数。代码、提示词、配置、模型或数据指纹变化则拒绝原运行恢复，需新run ID。

选择最好候选后可无LLM单独运行：

```bash
python -u "stage1+llm_v2/run.py" evaluate \
  --candidate "stage1+llm_v2/outputs/<run-id>/candidates/C0005.py" \
  --input turbine102.tsv --stage2-device cuda
```

87仅用于算法冻结后的最终测试，必须显式传 `--final-test`；不能将其成绩用于后续进化。

## 本次交付范围

没有导入旧候选，没有新增权重，没有自动运行收费搜索。复用的是独立复制到新目录的Stage2桥接/路线审核及KDE数据准备代码；旧目录不是运行依赖。仅做必要的语法、命令入口与协议连通检查，不把它们表述为完整回归或正式权重验收。
