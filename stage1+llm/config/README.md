# v3 配置

`search.json` 的相对路径从仓库根目录解析。真实密钥只填本机 `deepseek.local.json`，不要修改公开示例放密钥；私密文件不会随Git同步。模型ID须按账号实际可用值填写，离线测试不会核验供应商模型。

| 配置 | 默认及意义 |
|---|---|
| target_input | turbine102.tsv，真实搜索输入 |
| validation_inputs | []，可追加独立外部数据；不得包含87坐标 |
| protected_input | turbine87.tsv，仅用于搜索防混入检查 |
| data_dir | stage1+llm/outputs/prepared-v3，固定manifest |
| training_virtual_fields / validation_virtual_fields | 8 / 8，KDE拓扑数 |
| data_seed / validation_seed | 1234 / 1001234，不能重叠 |
| deployment | 半径7km、速度23m/s、风速6m/s、续航2460秒、悬停10秒、能耗比1.14、罗盘风向90度 |
| archive_size | 12，仅限制活跃父代池，不截断完整Pareto前沿 |
| initial_candidates / evaluation_budget | 3 / 40，独立初始轮次/总逻辑轮次，生成失败计入预算 |
| no_improvement_limit | 10，目标前沿无变化的停止轮数 |
| seed | 1234，父代及方向选择种子，不保证远端LLM逐字可复现 |
| candidate_timeout_seconds | 300，单候选单数据集的子进程墙钟超时 |
| heartbeat_seconds | 10，LLM等待、算法和可信评估心跳间隔 |
| ucb_decay / ucb_exploration | 0.9 / 0.7，历史方向评分衰减及探索强度 |
| lower_bound_nests | 1，仅保留旧配置兼容；不据此声明多目标全局最优 |
| direction_actions / diagnostic_direction_map | 方向说明及诊断映射；覆盖、归属、半径、续航修复 |

数据集冻结与物理配置分开记录；同一manifest可用于新物理实验，但不可在旧run中改变物理配置后继续搜索。恢复仅允许增大 `evaluation_budget`、通过 `--reset-stall` 重置停滞、修改日志心跳显示。其他设置改变必须新run；数据/种子改变还必须使用新data_dir。

`deepseek.local.json` 字段沿用 `base_url/model/api_key/temperature/timeout_seconds/retries/max_tokens`。请求尝试及SDK重试由传输钩子记录；响应token用量不是完整计费记录。日志脱敏不等同于磁盘加密。
