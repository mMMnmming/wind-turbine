# 单次贪心：整体路线优势 vs 后续累计收益

所有 Linux 命令在仓库根目录执行。正式训练只用102训练池和原固定72区域验证集；87不参与选模。
本实验保持网络参数、风况、动作约束和奖励总目标不变，不使用Critic、熵正则、最优路线标签或局部搜索。

## 训练什么

两组加载相同的 `finetune-edge-v2/control/best_greedy.pt` 权重及其活动训练池，分别重建Adam。
`trajectory` 使用整条路线的留一法优势；`reward_to_go` 使用当前动作至返航结束的后续收益，减去同图其他轨迹同一步的平均收益。
奖励拆分包含每段有向飞行、每个客户悬停、终端返航和原有安全奖惩，γ=1；不添加“选最近节点”奖励。
逐步损失先对每条轨迹有效步骤求和，再对图和轨迹平均，不按节点数额外归一化。

默认两组各50轮，种子1236，64图/批、每图8条网络采样、每轮100次更新；学习率3e-5余弦降至3e-6，梯度裁剪1。
每组重新从同一活动池开始，刷新频率与后续拓扑编号继承源模型，两组使用相同刷新序列。
训练仍然采样探索；每轮验证只生成一次贪心路线，绝不使用128候选选模。

## 后台正式运行（单行）

```bash
mkdir -p stage2/outputs/greedy-rtg-v1 && nohup python -u -m stage2.run greedy_rtg_compare --checkpoint stage2/outputs/finetune-edge-v2/control/best_greedy.pt --data-dir stage2/outputs/edge-v2-data --epochs 50 --output stage2/outputs/greedy-rtg-v1 > "stage2/outputs/greedy-rtg-v1/$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

这是顺序运行的50＋50轮，不是两组并发。首次运行必须使用新实验目录，已有实验需恢复。
源权重和数据目录不得删除或改动；不覆盖旧模型。可用 `tail -f stage2/outputs/greedy-rtg-v1/日期日志.log` 查看进度。

恢复使用完全相同参数，加 `--resume`（已完成组不重训）：

```bash
nohup python -u -m stage2.run greedy_rtg_compare --checkpoint stage2/outputs/finetune-edge-v2/control/best_greedy.pt --data-dir stage2/outputs/edge-v2-data --epochs 50 --output stage2/outputs/greedy-rtg-v1 --resume > "stage2/outputs/greedy-rtg-v1/$(date +%Y%m%d_%H%M%S)_resume.log" 2>&1 < /dev/null &
```

如需参考gap诊断，首次运行和恢复均加 `--reference-results 实际的验证集参考/reference_results.json`，不能传87参考。
参考不进入奖励、梯度或选模。缺少或未证明最优的参考显示N/A，不伪造gap。

## 查看验证与选模结果

```bash
cat stage2/outputs/greedy-rtg-v1/comparison.md
```

`comparison.json` 记录全部区域飞行总时间、可行率、归一化成本、任务规模分组和可用参考gap。
每组目录包含 `epoch0.pt`、`best.pt`、`best_greedy.pt`、`latest.pt`、`training_history.json` 和 `metrics.jsonl`。
两种best文件都是单次贪心最佳；latest保存优化器、调度器、活动池及随机状态用于恢复。
根目录 `selected.pt` 与 `selected_greedy.pt` 相同：先比可行率，再比平均归一化飞行时间；相同优先旧基线、其次trajectory。
全部微调退步时保留第0轮旧权重。最佳文件写入中断可从latest修复。

日志的 `reward` 是训练采样路线奖励；`feasible` 是验证可行率；`val best` 与 `greedy` 在此入口均指同一次贪心成绩，不是候选择优；`best greedy epoch` 为最佳轮次。
JSON额外记录有效步骤优势标准差、路线多样性、梯度范数、裁剪比例、策略熵（仅监测，不加熵奖励）。
微小单种子收益不等于统计显著；只有RTG不降低可行率且优于trajectory的贪心验证成本，才有本次适配有效的证据。

重新导出选定模型的完整验证路线（不训练）：

```bash
python -u -m stage2.run evaluate --regions-json stage2/outputs/edge-v2-data/validation_regions.json --checkpoint stage2/outputs/greedy-rtg-v1/selected_greedy.pt --decode-strategy greedy --decode-samples 1 --evaluation-output stage2/outputs/greedy-rtg-v1-validation
```

## 验证选定后，87最终测试与最优解比较

继续使用已有冻结87区域，不重新部署：

```bash
python -u -m stage2.run evaluate --regions-json "stage2/Branch and Bound/outputs/optimal87/regions.json" --checkpoint stage2/outputs/greedy-rtg-v1/selected_greedy.pt --decode-strategy greedy --decode-samples 1 --evaluation-output stage2/outputs/model87-greedy-rtg-v1
python "stage2/Branch and Bound/compare.py" --model-results stage2/outputs/model87-greedy-rtg-v1/model_metrics.json --reference-results "stage2/Branch and Bound/outputs/optimal87/reference_results.json" --output "stage2/Branch and Bound/outputs/comparison-greedy-rtg-v1"
```

不要根据87成绩反复选组或调参数。功能测试通过不代表模型更优；本地2轮流程检查使用微型模型，不是用户正式模型成绩。
