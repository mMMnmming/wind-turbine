# Stage 1: Walney-189 UAV 巢部署与区域划分

本目录复现论文的核心部署/分区部分：先以每台风机为候选 UAV 巢构建可行组，再删除冗余候选巢，最后将每台风机唯一分配给最近的保留巢。

## 范围

- 输入仅为 `../turbine189.tsv`，格式固定为 `风机 ID<TAB>x<TAB>y`。
- 输出只覆盖巢部署与区域划分；不包含后续 DRL 路径规划、87/102 子集对照或 500 台扩展实验。
- 坐标按 km 处理。`turbine189.tsv` 的约 14.5 km × 30.2 km 跨度与论文使用的 7 km 作业通信半径一致。

## 约束与建模

算法对每个最终巢逐项验证：

1. 每台风机恰好分配给一个巢；巢坐标必须来自风机坐标。
2. 巢到每台分配风机的距离不超过有效服务半径：`min(通信半径, 风中往返最大半径)`。
3. 以最近邻闭环路线估计飞行时间；`飞行时间 + 1.14 × 悬停时间` 不得超过最大续航。
4. 候选巢只在其全部成员仍由其他候选巢覆盖时删除；重叠覆盖按巢距离最小、ID 字典序最小分配。

默认配置位于 `config.json`：通信半径 7 km、风速 6 m/s、最大水平速度 23 m/s、续航 41 min、每台悬停 10 s、悬停能耗比例 1.14。部署使用保守逆风地速 `23 - 6 = 17 m/s`。配置中的参数都可替换，不需要修改算法代码。

论文正文可直接确认 7 km、6 m/s、10 s 与 1.14；表 III 的图形化排版未能由 PDF 文本层可靠导出，因此 23 m/s 与 41 min 被作为可核验默认值而非不可修改的论文常量。功率、电池 Wh 未被擅自补入模型。

## 运行

在项目根目录执行：

```powershell
pip install -r stage1\requirements.txt
python stage1\run.py
```

可替换输入、配置和输出目录：

```powershell
python stage1\run.py --input turbine189.tsv --config stage1\config.json --output stage1\results
```

默认行为与参考脚本一致：保存 PNG 后打开 Matplotlib 图窗。若只需要文件、不要打开图窗，使用 `--no-show`：

```powershell
python stage1\run.py --no-show
```

## 输出

- `results/summary.json`：完整配置、巢列表、组内最近邻闭环与全部约束余量。
- `results/assignments.csv`：每台风机的唯一巢归属。
- `results/nests.csv`：每个巢的规模、航程、等效能耗时间和余量。
- `results/iteration_log.csv`：从第 0 次初始化到最终收敛的逐轮删巢日志，含每轮删除的巢、活跃巢数和覆盖冗余度。
- `results/deployment.png`：按 `遗传对比试验.py` 的 `plot_solution` 原样复用：Times New Roman、字号 18、`figsize=(6, 7.5)`、深蓝色 `s=50` 风机、带黑边的红色 `s=100` 方形 UAV 巢、蓝色 `alpha=0.5` 空心覆盖圆、字号 16 的右上角图例、虚线网格、等比例坐标与相同边距。坐标上下限仅改为根据 Walney-189 全量坐标自动计算，避免参考脚本的示例 X 轴范围裁掉数据。
- `results/iteration_process.png`：最终删巢状态，采用与最终选址图完全相同的绘图形式。
- `results/iteration_step_000.png`、`iteration_step_030.png`、`iteration_step_060.png`、`iteration_step_090.png`：删巢过程的四个快照；每一张都是独立的 `6×7.5` 选址图，而非压缩子图或收敛线。

运行时控制台会同步逐轮打印 `Iteration / removed / active nests`。JSON 与 CSV 内容按固定顺序写入；PNG 使用固定 Matplotlib 版本以便稳定复现视觉样式。

## 测试

```powershell
python -m unittest discover -s stage1\tests -v
```

测试覆盖 TSV 输入校验、半径边界、续航边界、冗余巢删除、最近巢并列规则、确定性，以及 Walney-189 全量约束验证。
