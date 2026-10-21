# Stage2：旧模型保护下的两组微调

在仓库根目录、原训练环境运行。当前 `train-edge-v2/best.pt` 是只读基线，不会覆盖；不加入 Critic、教师、最优路线标签，不改变网络结构、奖励、物理约束或 Stage1。

## 1. 两组到底做什么

| 项目 | control（A组） | entropy（B组） |
|---|---|---|
| 初始权重 | 同一个旧 best.pt | 同一个旧 best.pt |
| 训练轮数 | 50 | 50 |
| 损失 | 原 REINFORCE | 原损失减去 β×有效步骤平均策略熵 |
| β | 0 | 第1轮0.01，线性降至第50轮0 |

重建 Adam；学习率3e-5余弦降至3e-6，默认种子1236。批次、每图8轨迹、每轮100次更新、奖励和128候选验证等继承父checkpoint。两组都加载父checkpoint的活动训练池，继承后续KDE字段流，微调本地每5轮替换32个风场。随机流重置为相同种子；顺序执行两组，避免抢占GPU。熵正则仅改变损失，不更改验证解码温度或候选预算。

原 `resume` 模式是精确续训，不能用于更改算法参数；此处必须使用新 `finetune_compare` 模式。不需要传 `--config`，架构、奖励、数据和解码口径从checkpoint继承。仅加载自己信任的checkpoint（PyTorch文件包含序列化训练状态）。

## 2. 一条命令后台运行（共50＋50轮）

需要保留服务器上原模型及完整的 `stage2/outputs/edge-v2-data/`。不需要重建数据，不会读取87。输出目录必须未使用过；可以先 `mkdir`，目录内只有日志不影响启动。

```bash
mkdir -p stage2/outputs/finetune-edge-v2 && nohup python -u -m stage2.run finetune_compare --checkpoint stage2/outputs/train-edge-v2/best.pt --data-dir stage2/outputs/edge-v2-data --epochs 50 --output stage2/outputs/finetune-edge-v2 > "stage2/outputs/finetune-edge-v2/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

如需诊断加权最优性gap，可以在**首次启动时**于 `--epochs 50` 后加入：

```text
--reference-results "stage2/Branch and Bound/outputs/validation-edge-v2/reference_results.json"
```

参考只用于报告，不参与训练损失或选模。无参考仍能正常运行。报告的 `weighted_optimality_gap` 是小数比例（乘100为百分比），只覆盖模型可行且参考已证明最优的正成本区域；`optimality_gap_count` 是有效区域数。不要与旧的逐区域平均 `mean_optimality_gap` 混淆。开始后不得修改参考文件内容，否则恢复身份核验会拒绝；完成参考恢复求解后再启动实验更稳妥。

## 3. 中断恢复

确认原进程已停止，再用相同参数加 `--resume`。不要同时向同一输出目录启动两个进程。

```bash
mkdir -p stage2/outputs/finetune-edge-v2 && nohup python -u -m stage2.run finetune_compare --checkpoint stage2/outputs/train-edge-v2/best.pt --data-dir stage2/outputs/edge-v2-data --epochs 50 --output stage2/outputs/finetune-edge-v2 --resume > "stage2/outputs/finetune-edge-v2/$(date +%Y%m%d_%H%M%S)_resume.log" 2>&1 < /dev/null &
```

首次若加过 `--reference-results`，恢复时也必须加同一份不变参考。已完成的A组不会重训，B组从最近完整轮次恢复；中断在一轮内部，该轮重跑。必须保留实验目录完整文件和源checkpoint。配置、数据、参考、源权重哈希必须一致；不支持在同一实验里把50轮临时改成100轮。

## 4. 看什么结果

根目录 `comparison.md` / `comparison.json` 对比旧模型、A组、B组的验证成绩，并给出 `selected_sampling` / `selected_greedy`。严格按可行率优先、平均归一化飞行时间其次选优，完全相同保留旧模型，其次A组；不按87成绩选。

- `selected.pt` / `selected_sampling.pt`：旧模型、A、B中验证128候选最好的权重。
- `selected_greedy.pt`：验证单次贪心最好的权重，未必与上者相同。
- 每组：`epoch0.pt`、`latest.pt`、`best_sampling.pt`、`best_greedy.pt`、`best.pt`、`training_history.json`、`metrics.jsonl`、`pool_history/`。
- `best.pt` 与 `best_sampling.pt` 内容等价；`latest.pt` 才用于本实验内部恢复。开始时评估并保存第0轮；若全部微调退步，最佳模型继续保留第0轮权重，不会强行采用新模型。
- `latest.pt` 内嵌两套最佳权重和选模记录；恢复会修复中断留下的过时最佳文件。最佳模型文件用于评估或作为后续微调起点，不用于普通 `resume` 模式。

日志增加beta、熵损失贡献、优势标准差、路线多样性和两类最佳轮次。可行率百分比是按原程序浮点指标展示；并不保证未见实例安全，更不保证全局最优。

## 5. 验证选定后，87只做最终报告

```bash
mkdir -p stage2/outputs/model87-finetune && nohup python -u -m stage2.run evaluate --regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json" --checkpoint stage2/outputs/finetune-edge-v2/selected.pt --decode-strategy best_of_n --decode-samples 128 --decode-chunk-size 32 --evaluation-seed 1234 --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --evaluation-output stage2/outputs/model87-finetune > "stage2/outputs/model87-finetune/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

```bash
python "stage2/Branch and Bound/compare.py" --model-results stage2/outputs/model87-finetune/model_metrics.json --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --output "stage2/Branch and Bound/outputs/comparison-finetune"
```

以上也保存同一权重的 `greedy_metrics.json`。若要测**独立选出的贪心最佳权重**，另外运行：

```bash
python -u -m stage2.run evaluate --regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json" --checkpoint stage2/outputs/finetune-edge-v2/selected_greedy.pt --decode-strategy greedy --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --evaluation-output stage2/outputs/model87-finetune-greedy
python "stage2/Branch and Bound/compare.py" --model-results stage2/outputs/model87-finetune-greedy/model_metrics.json --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --output "stage2/Branch and Bound/outputs/comparison-finetune-greedy"
```

不增加候选预算掩盖退步；单种子对照只能说明本次实验表现，不能据此宣称统计显著提升。
