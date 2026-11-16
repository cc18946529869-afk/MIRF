# 原论文协议的来源

本文件只记录这次核对读到的材料。三层分开：论文写明的事实、仓库代码或本地表能够验证的事实、还不能下的结论。

## 读过的材料

- 预印本 bioRxiv 2023.11.16.565910，2025-03-11 的版本。正文、方法 4.2、4.2.1、4.2.2、4.2.3、4.3，以及 Fig. 4 的图注。正式期刊是 Nature Machine Intelligence，`s42256-025-01103-w`。这次没有拿到付费全文的逐段对照，正文和公式以下面这份预印本为准。
- 同一篇的补充材料 PDF。补充图 11 写明 TEM-1 对氨苄青霉素和头孢噻肟的效应相关性很弱。补充材料没有列出 16 条 MIC 的逐行预测，也没有列出六个 checkpoint 的文件名。
- Weinreich、Delaney、DePristo、Hartl，Science 2006，Table 1 里不带启动子突变 g4205a 的头孢噻肟 MIC 列。
- 本分支父提交 `1fb9dc0f78758ccab72216ef65a00234bd9008be` 里的 `baseline.py`、`dirichlet_ppo.py`、`env.py`、`muformer.py`、`landscape.py`，以及 vendored PPO 的默认参数。
- 本地表 `CFX_test_from_pan.tsv`、`esbl_sequences.csv`，以及 ProteinGym v0.1 划分 `BLAT_ECOLX_Stiffler_2015_{train,valid,test}.tsv`。

## 论文写明的事实

微调数据是 Stiffler 等人 2015 年在头孢噻肟下测量的大约 5,000 个 TEM-1 饱和单点突变。评估集是整理过的 105 个 ESBL 和 52 个非 ESBL。Fig. 4d 用 Weinreich 等人的 16 个 TEM-1 变体，其中 11 个是高阶突变，标签是 MIC。正文把这次相关写成 Spearman ρ = 0.94。Fig. 4 图注把同一张图写成 Spearman ρ = 0.92。这两个数都在预印本里，不能并成一个数。

搜索被定义成一个 MDP。状态是当前突变序列。动作是一次单氨基酸替换，先选位点，再选氨基酸。智能体采样到预先定下的 horizon 才结束一条轨迹，终点是相对野生型的多点突变体。奖励由 μFormer 给出。方法 4.2.1 把每一步的优势写成 \(A_t=S(x)-v_\phi\)，\(S(x)\) 是生成突变体的 μFormer 分数。两个价值网络分别估计给定状态之后的回报。探索分布是 \(P=(1-\varepsilon)\pi+\varepsilon\eta\)，\(\varepsilon=0.25\)，\(\eta\sim\mathrm{Dir}(0.03)\)。更新时的概率比用混合分布，而不是纯策略。

TEM-1 湿实验的目标写的是 2–3 个氨基酸替换。六个不同随机种子训练的 μFormer 各做一次 μSearch。每次搜索的阈值是该模型自己对已知 ESBL 的预测中位数。六次合计大约筛了一百万条序列，汇总出 1,200 个以上超过各自阈值的唯一突变体。每条候选由六个模型重新评分，在各自搜索里取相对排名，再把六个排名加总。按这个聚合排名取前 200 个做湿实验。湿实验只有一轮。47/200（23.5%）在 0.2 µg/ml 头孢噻肟上的生长高于野生型；随机 2–3 突变对照是 12%。另外放入了 50 个随机 2–3 突变体，以及 15 个覆盖 A40G、E102K、M180T、G236S、阶数为 1–4 的已有 MIC 变体。一个设计出的双突变 G236S;T261V 超过了已知四突变体。

## 代码和本地表能够验证的事实

`examples/baseline.py` 里 `musearch_gt` 和 `musearch` 都把 `horizon` 设为 5，把 `score_threshold` 设为负无穷。`env.py` 的 `step` 每一步都写入所选位点和氨基酸，计步加一；只有 `cur_mutations == horizon` 时才调用 `get_fitness`，否则奖励是 0。重复位点和写成当前氨基酸都没有被拒绝。

`MuformerLandscape` 只加载 `muformer-l-BLAT_ECOLX_Ranganathan2015_CFX.pt` 这一个文件。初始化时读取 `esbl_sequences.csv`，计算 ESBL 分数的中位数和标准差。`predict` 只有在 `normalize=True` 时才用这个中位数。搜索环境调用的是未归一化分数。因此这份代码把 ESBL 中位数算出来了，但发布的基准搜索不用它做候选阈值。

`muformer.py` 文件头的示例命令写了六个 checkpoint 名字。它们的差别是 rank 或 Spearman 目标、weight decay，以及其中一个用 UR90 编码器。文件名里没有随机种子编号。`MuformerLandscape` 不加载这六个名字。`MuformerLandscapeList` 会把一个用空格分开的路径字符串建成多个景观；`baseline.py` 的 TEM-1 入口不走这个类。

vendored PPO 的默认值是 \(\gamma=0.99\)、\(\lambda=0.95\)。`dirichlet_ppo.py` 构造 PPO 时没有改这两个数。当前正式基线的更新概率是边际 \(0.75\pi+0.25/K\)；采样仍是 \(0.75\pi+0.25\mathrm{Dir}(0.03)\)。这和论文里“更新也使用同一次 Dirichlet 混合”不是同一条式子。

本地 `CFX_test_from_pan.tsv` 有 16 行。突变名和 MIC 与 Weinreich 2006 Table 1、不带 g4205a 的那一列一致。论文编号 A40G、E102K、M180T、G236S 比 Ambler 编号少 2。这 16 条里阶数是 0 有 1 条、1 有 4 条、2 有 6 条、3 有 4 条、4 有 1 条，最高是 4。

本地 ProteinGym 划分 `BLAT_ECOLX_Stiffler_2015` 的 train、valid、test 分别是 3497、500、999 条，合计 4996。相对这份 286 位野生型，全部是 1 个氨基酸差异，三份划分没有共同序列。本地没有找到 `mmc4-cfx.aligned.tsv`。

`esbl_sequences.csv` 有 104 条，不是论文写的 105。其中 101 条长度 286，3 条长度 285。已完成运行的初始化是 105 次评分，等于这 104 条再加一次野生型。

## 还不能下的结论

不能把这份 checkpoint 证明成 Fig. 4d 那一张图所用的模型，也不能证明它是六个随机种子中的某一个。保存的验证相关 0.566947 是训练日志，不是 0.94 或 0.92。

不能从 `baseline.py` 推出湿实验搜索用了显式的 2–3 阶过滤器。论文把 2–3 个替换写成实验目标；发布基准的 horizon 是 5。`landscape.py` 里有被注释掉的 `horizon_3` 文件名，那些行没有参与运行。

预印本图的文字抽取里出现过 12374 这个数，但没有和 TEM-1 头孢噻肟写在同一句。不能把它当成当前 CFX checkpoint 在 7–10 个突变上的测试集。Gonzalez 与 Ostermeier 2019 在方法里是生长实验方案的引用，和氨苄青霉素双突变库不是同一件事。补充图 11 已经说明这两种抗生素的效应不能互相代替。

Figshare 文章 30227557 的 API 这次返回 403，文章页正文为空。本次没有从 Figshare 重新拿到文件清单或官方校验值。
