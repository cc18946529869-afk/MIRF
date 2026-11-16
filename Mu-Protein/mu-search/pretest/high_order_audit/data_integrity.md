# 六次运行的数据核验

原始文件没有修改。下面只记录读到的内容。

WT 六次一致，长度 286。`role=start` 的分数是 -0.072684407。

六次目录里没有 `action_trace.csv`。`time.json` 的键是 landscape_seconds、method、search_seconds、seed，其中没有 horizon。终点汉明距离不能反推空编辑、重复编辑或回退。

μSearch 三次的 ppo_schedule 相同：rollouts=3，episodes_per_update=1666，n_steps=8330，total_timesteps=24990，terminal_scores=4998。n_steps/episodes_per_update=5，这是产物里能直接读到的每条轨迹编辑次数。AdaLead 的 metadata 没有 ppo_schedule，也没有 horizon。

| 运行 | 候选行 | 候选唯一序列 | reward | CNN | gradient | 索引 | 初始化长度 | 返回重评分 | 全通道重复 | 搜索分钟 | 问题 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| musearch_gt/seed0 | 4998 | 4998 | 4998 | 0 | 0 | 连续 | {286: 102, 285: 3} | 100 | 101 | 30.3 | 无 |
| musearch_gt/seed1 | 4998 | 4998 | 4998 | 0 | 0 | 连续 | {286: 102, 285: 3} | 100 | 101 | 30.5 | 无 |
| musearch_gt/seed2 | 4998 | 4998 | 4998 | 0 | 0 | 连续 | {286: 102, 285: 3} | 100 | 101 | 28.3 | 无 |
| adalead_gt/seed0 | 4998 | 3599 | 4998 | 0 | 0 | 连续 | {286: 102, 285: 3} | 99 | 1502 | 6.4 | 无 |
| adalead_gt/seed1 | 4998 | 3517 | 4998 | 0 | 0 | 连续 | {286: 102, 285: 3} | 99 | 1583 | 6.4 | 无 |
| adalead_gt/seed2 | 4998 | 3498 | 4998 | 0 | 0 | 连续 | {286: 102, 285: 3} | 99 | 1603 | 6.3 | 无 |

候选行都是 4998，分数都有限，CNN 和 gradient 都是 0，`candidate_index` 都是 1 到 4998。候选序列长度都是 286，没有空序列，也没有标准氨基酸以外的字符。同一序列没有出现两个不同的分数。

初始化共 105 条，六次运行的长度分布相同：3 条长度为 285、102 条长度为 286。长度不是 286 的初始化序列有 3 条，六次是否为同一组：是。这是初始化表里的既有差异，候选表没有因此缺行。

μSearch 候选去重：musearch_gt/seed0 候选唯一序列 4998，全通道重复 101；musearch_gt/seed1 候选唯一序列 4998，全通道重复 101；musearch_gt/seed2 候选唯一序列 4998，全通道重复 101。三次候选唯一序列都是 4998，全通道重复发生在候选轴以外。AdaLead 候选轴上的重复调用：adalead_gt/seed0 重复调用 1399，其中非 WT 0；adalead_gt/seed1 重复调用 1481，其中非 WT 2；adalead_gt/seed2 重复调用 1500，其中非 WT 1。返回重评分：musearch_gt/seed0 返回重评分 100，musearch_gt/seed1 返回重评分 100，musearch_gt/seed2 返回重评分 100，adalead_gt/seed0 返回重评分 99，adalead_gt/seed1 返回重评分 99，adalead_gt/seed2 返回重评分 99。这些差异不改变 4998 条候选行。

`seeds.json` 写明同一整数只是把四个生成器设成该整数，两种方法的随机数流并不相同。

查询口径按每一次候选调用计数。唯一序列口径对每条候选序列只保留第一次分数。重复调用不另算成新的突变体。
