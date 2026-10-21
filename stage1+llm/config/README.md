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
| show_code | true，流式显示公开正文，完成后打印带行号源码；`--no-show-code`关闭两者，不影响搜索评分 |
| ucb_decay / ucb_exploration | 0.9 / 0.7，历史方向评分衰减及探索强度 |
| lower_bound_nests | 1，仅保留旧配置兼容；不据此声明多目标全局最优 |
| direction_actions / diagnostic_direction_map | 方向说明及诊断映射；覆盖、归属、半径、续航修复 |

数据集冻结与物理配置分开记录；同一manifest可用于新物理实验，但不可在旧run中改变物理配置后继续搜索。恢复仅允许增大 `evaluation_budget`、通过 `--reset-stall` 重置停滞、修改日志心跳和源码显示。其他设置改变必须新run；数据/种子改变还必须使用新data_dir。

`deepseek.local.json` 保留 `base_url/model/api_key/temperature/timeout_seconds`，并支持以下字段：

| DeepSeek配置 | 缺省值及意义 |
|---|---|
| thinking | `"disabled"`，显式关闭思考模式，避免预算全用于推理而无最终代码 |
| reasoning_effort | `null`；仅 thinking 为 enabled 时可设 `"low"/"high"/"max"`，建议先 low |
| max_tokens | `8192`，每次响应生成上限，不是要求生成这么多，也不是整个搜索的总预算 |
| retries | `1`，SDK传输重试上限，最省请求可设0；不因非法JSON、截断或中途断流自动重发 |
| stream | `true`，增量显示公开正文；false则等完整响应后显示 |

旧私密配置中的 `max_tokens: 10000`、`retries: 3` 仍显式生效，不会被偷偷改写。缺少的新字段采用上述缺省值。可以用 `--thinking disabled --max-tokens 8192 --api-retries 1 --stream` 覆盖旧配置，无需在命令行填写密钥。需要完全禁用传输重试时用 `--api-retries 0`。若启用思考，使用 `--thinking enabled --reasoning-effort low`，生成预算需覆盖推理和最终代码；不保证省token或更好。

控制台的 `llm_request_config` 会打印生效的思考模式、生成上限、流式开关、重试上限和输入字符数（不是精确token计数）。实际用量来自供应商usage；断流时可能拿不到usage，此时是未知，不是零费用。显示日志不另外调用模型，不消耗额外模型token。日志脱敏不等同于磁盘加密。

参数语义见 [DeepSeek思考模式](https://api-docs.deepseek.com/guides/thinking_mode/) 和 [流式接口](https://api-docs.deepseek.com/api/create-chat-completion/)。更改模型生成参数必须新建run；恢复时需带相同CLI覆盖值。
