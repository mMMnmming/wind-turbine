你是风机巡检机巢部署搜索算法的代码设计者。输出完整同步Python策略模块，不输出某个固定实例的答案。第一代从零设计，不依赖C0003或任何历史源码。后续根据父代完整代码与真实执行反馈进化算法。

硬边界：框架在每个实例、每代代码运行前构造并验证“每台风机各设一巢”初解。你的策略提出完整候选部署；框架独立验证全部覆盖、合法机巢、自身归属、7km等配置服务半径，用冻结Stage2单次贪心生成路线并审核实际含风闭环、返航、非巢任务悬停和续航。你不能修改模型、风况、约束、比较规则和预算，不能自己宣称可行性或成绩。阵风仅按现有Stage2特征语义处理，不代表随机阵风鲁棒安全。

内层严格字典序最小化：[机巢数,全场总飞行秒,最长区域飞行秒]。框架只接受严格改善并保存；不接受更差的当前状态。因此不要声称实现需要接受劣解的模拟退火。跨实例全部可行且代码正常后比较平均巢数、平均总飞行秒、全体最长区域秒；运行时间独立报告，不改变排序。

每个实例都重新从单风机单巢起步；子代只继承父代代码和经验，不继承部署。你可以自由设计完整构造和搜索机制，例如区域合并、撤巢联合重分配、多区域联合修改、迁巢、固定巢数搬移/交换、多候选排序、变邻域与多种策略调度。每次回调只返回一个完整候选，后续根据真实反馈决定下一步。不要求固定减巢/迁巢循环。程序自己完成大量搜索，不在内层再次请求LLM。策略必须在有限步骤内返回，避免在一次回调里穷举所有部署。

源码必须恰好定义以下两个顶层入口，可包含全部辅助函数：
def initialize(problem, initial_solution, seed):
    # 返回JSON可序列化的state
def propose(problem, current_solution, state, feedback):
    # 返回恰好四字段：
    return {"assignments": 完整dict或None, "state": 新状态, "phase": "简短阶段名称", "done": False}

problem是dict：
- turbines: [{"identifier":str,"x_km":float,"y_km":float}, ...]，全部允许设巢，坐标km。
- config: communication_radius_km,max_horizontal_speed_mps,wind_speed_mps,wind_direction_degrees,max_endurance_seconds,hover_seconds_per_turbine,hover_to_flight_energy_ratio,coordinate_unit。
- environment: Stage2实际固定环境，含gust_speed_mps。
- time_matrix_ids: 有向航段矩阵行列的ID顺序。
- travel_time_seconds: 同Stage2风模型生成的二维有向时间矩阵，可用于排序和构造，不能替代Stage2实际路线审核。

initial_solution/current_solution是dict：assignments完整映射、nest_ids、metrics、objectives。current_solution始终是框架保存的最好解。你修改收到的副本不会改变框架状态，必须返回候选才会评价。所有输入用字典访问，不用属性访问。

feedback包含：kind(initial/evaluation/duplicate/no_proposal)、accepted、feasible、objectives、diagnostics、longest_region、tightest_region；可选字段用get读取。remaining包含剩余seconds/proposals/stage2_inferences/stagnation_remaining。未被接受时current_solution不变，你需要用state推进候选游标，不能反复提出同一方案。done=True表示策略声明当前搜索结束，不代表已证明全局最优；允许在最后一个候选一起返回done。assignments=None表示本次不提交，仍消耗一次动作。

部署完整映射：每个风机ID恰好一个键，值是启用机巢ID；每个启用机巢必须归属自身。撤巢后原巢对应风机仍要被分配，不能删键。迁巢后旧巢任务和新巢原区域都要妥善处理。可直接构造完整dict；也可用可选纯结构模块：
from candidate_ops import merge, relocate, remove, reassign
merge(a,source,target)返回合并后的完整映射，包括source自身任务。
relocate(a,source,target)返回完整映射；target必须是尚未启用的风机。
remove(a,source,replacements)：replacements的键必须恰好等于source全部区域成员（包括source），值为其他启用巢。
reassign(a,updates)：仅移动非巢任务到现有巢，返回完整映射。
这些工具不修改输入，不选目标、不搜索、不评价、不修复。例：a={'a':'a','b':'a','c':'c'}；merge(a,'a','c')得到{'a':'c','b':'c','c':'c'}。

只允许import math,random,itertools,collections,typing,candidate_ops；显式导入辅助模块。可用常见数值/容器内建函数及Exception,ValueError,KeyError,RuntimeError,TypeError,IndexError,StopIteration,ZeroDivisionError；不提供print、getattr、hasattr、type、文件/网络/进程/动态执行/反射/类/异步/global/nonlocal。禁止私有属性和双下划线。random用random.Random(seed)初始化；若需要随机状态，将其转成JSON兼容状态（或在state保存随机种子/计数）。state只能含JSON数据，不存函数、迭代器、集合、Random对象。每次返回数据小于4MB。不要输出真实密钥或未执行的成功声明。禁止使用87测试数据。

只输出一个JSON对象，恰好六个非空字符串字段：algorithm_idea,change_hypothesis,expected_improvement,change_summary,direction_id,algorithm_code。direction_id须与本轮要求一致。algorithm_code是完整源码，其他说明各不超过120字。不要Markdown、补丁、运行日志或额外字段。
