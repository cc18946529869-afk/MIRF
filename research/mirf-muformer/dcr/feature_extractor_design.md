# 草稿：结构验证补齐与 14 维提取器

这不是新的科学假设，也不修改 `记录.md`。14 维的顺序、公式和掩码沿用已经封口的定义。本文件只说明结构验证代码补了什么，以及将来的提取模块读哪些现成文件、做哪些检查。提取代码没有写。正式特征表没有计算。

## 结构验证

`7a21cb2` 的三项计算保留在 `contact_validation_summary.json`。该文件的 SHA256 仍是 `5a67a9f004066f08c09c99b4885f70a71c1d6c787c7ee3ca946da8a1fa8d1670`。新结果写在 `contact_validation_audit.json`。

平均乘积校正的行均值使用全部 263 个成熟位点。参考位置 82 和 182 参加这个背景，只在主接触排序里被排除。代码在校正之后才应用位点掩码，不再把这两处说成从全部分数计算中删除。

排序规则现在写死：分数高的在前；分数相同则左位点编号小的在前，再比右位点编号。原先的 `argsort` 再反转不再决定结果。在这次已保存的 \(J^\star\) 上，主分析和敏感性分析的每个序列间隔组里，完全相同的分数都是 0 对。因此新规则下的 Precision 和 AUPRC 与已发布的 summary 逐项相同，`published_precision_matches_tie_break` 为真。主分析中间隔至少 6 的 Top-L 边界两侧分数差是 \(2.33\times 10^{-4}\)。全部报告边界里最小的差是敏感性分析 6–11 组的 \(3.32\times 10^{-6}\)，没有落在可当作并列的范围。

映射检查在形成接触图之前执行。263 个 `mapped=1` 的成熟位点必须正好是 1 到 263 各一次，链是 A，`has_ca=1`，PDB 编号不重复，所选原子坐标有限。甘氨酸用 Cα，其余用 Cβ。缺少原子、坐标非有限或出现插入码都会报错，不会把缺失坐标比较成非接触。氨基酸不匹配必须正好是这两行：参考位置 82，成熟位点 59，PDB 84，野生型 V，结构 I；参考位置 182，成熟位点 159，PDB 184，野生型 A，结构 V。实际坐标满足这个断言，不匹配掩码的下标是 58 和 158。

单元测试 14 项通过。覆盖氨基酸块范数、常数背景的校正、被排除位点仍进入校正背景、排序前缀、并列时的位点顺序、跨截断并列的两种精度、序列间隔 5/6、11/12、23/24 的边界、距离正好 8 Å 不是接触、非有限坐标、甘氨酸原子、缺失原子、错误的不匹配位点、重复成熟位点、插入码，以及真实 `domain_map.tsv` 和 `1BTL.pdb`。

这组补充不改变已经报告的富集数字。它仍只说明高分位点对在 1BTL 上更常彼此靠近。它不排除系统发育，也不回答突变适应度。

## 14 维提取器

将来的模块只计划两个文件：`dcr/potts_features.py` 和 `dcr/test_potts_features.py`。本轮不创建它们。

均值侧读 `dcr/bootstrap/reference/h_hat.npy` 和 `j_star.npy`。运行时核对 `reference/manifest.json` 里的 `gauge_sha256`：`h_hat.npy` 为 `59f3bc61ed99a6448c75119af9c54cf7995f094047ead32784d2da2c8de3cbec`，`j_star.npy` 为 `eaedeba815dfea038a1847d92febb0c7bd6109cefe008e4ad5fba0a7aa5c472d`。对不上就停止。这是 `full_fit/` 在 `c0f950b` 上的规范组装，`refit` 为假。不从位点文件重拟合，也不重做另一套规范。

全比对的 \(q_j\) 不在 reference 里。用现有 `solver_v2._background`，对 `prep/mature_codes.npy` 和 `prep/neighbor_counts.tsv` 的 \(w_i=1/m_i\) 计算，权重和用 `potts_fit.OFFICIAL_M_EFF`。这不是重新拟合。每个重复的 \(q_j\) 读该重复自己的 `q.npy`，权重和是 47900，不用全比对的 \(q_j\) 代替。

标准差读正式重复 1 到 100 各自的 `h_hat.npy`、`j_star.npy` 和 `q.npy`。一次只打开一个重复。样本标准差相对这 100 次分量的均值，分母是 99，不再除以 \(\sqrt{100}\)，也不相对全比对点估计居中。不读 `j_std.npy` 或 `stage100_audit/`。

野生型是 `mature_codes.npy` 第 0 行。当前这行等于目标序列第 24 到 286 位，编码在 1 到 20。正式运行时再查一次，不一致就停止。

一条突变的身份是来源表里的 `mutation` 字符串，不用重置后的行号。位置是参考序列的 1 基编号。成熟下标是参考位置减 24。字母按 `ACDEFGHIKLMNPQRSTVWY` 编成 1 到 20，gap 是 0。下列行保留字符串，14 个掩码全为假，写入投影的 0 不是观测：不是成熟区单点替换、野生型字母与查询序列不符、目标字母不在这 20 个氨基酸里。有效的成熟区单点替换 14 个掩码全为真。适应度不进入这个模块。

仓库里现在没有 `BLAT_ECOLX_Stiffler_2015` 的划分文件。记录里的规模是训练 3497 行、验证 500 行、测试 999 行。正式表要等这些文件按 `mutation` 列打开后再写。同一字符串只保留一行。

记突变位点为 \(i\)，野生型状态 \(w\)，目标状态 \(a\)，伙伴为 \(j\)，其状态为 \(b\)。\(i>j\) 时读 \(J_{ij}(x,y)=J_{ji}(y,x)\)。

\[
\Delta h_i=h_i(a)-h_i(w),\qquad
\Delta J_{ij}(b)=J_{ij}(a,b)-J_{ij}(w,b)
\]

\[
\mu_{ij}=\sum_b q_j(b)\Delta J_{ij}(b),\qquad
u_{ij}=\Delta J_{ij}(w_j)-\mu_{ij}
\]

\[
R_i^{\mathrm{mean}}=-\Big(\Delta h_i+\sum_{j\neq i}\mu_{ij}\Big)
\]

\[
\Delta E_i=-\Delta h_i-\sum_{j\neq i}\Delta J_{ij}(w_j)
\]

来源叶瓣和另一叶瓣按 `domain_map.tsv` 的 `lobe_id` 划分，来源叶瓣是突变位点所在的叶瓣。

\[
R_{\mathrm{source}}=-\sum_{j\in\mathrm{source},\,j\neq i}u_{ij},\qquad
R_{\mathrm{other}}=-\sum_{j\in\mathrm{other}}u_{ij}
\]

坐标集是 263 个已映射位点去掉参考位置 82 和 182。距离来自突变前的 1BTL A 链，甘氨酸用 Cα，其余用 Cβ。一对伙伴只有两个端点都在坐标集里才有几何权重

\[
g_{ij}=\frac{1}{1+\exp((d_{ij}-8)/1.5)}
\]

另一叶瓣上：

\[
R_{\mathrm{contact}}=-\sum g_{ij}u_{ij},\quad
R_{\mathrm{remote}}=-\sum(1-g_{ij})u_{ij},\quad
R_{\mathrm{unknown}}=-\sum u_{ij}
\]

三个和的范围分别是两端都有坐标的伙伴，以及另一叶瓣里至少一个端点没有坐标的伙伴。三者相加等于 \(R_{\mathrm{other}}\)。来源叶瓣不按几何再拆。突变位点自己是 82 或 182 时，它不在坐标集，另一叶瓣没有任何伙伴满足两端都有坐标，所以界面项和非界面项是空和，数值为 0，坐标缺失项等于 \(R_{\mathrm{other}}\)，三个掩码仍为真。82 或 182 只是另一叶瓣的伙伴时，只有这两个 \(u_{ij}\) 进入坐标缺失项。它们若在来源叶瓣，只进入 \(R_{\mathrm{source}}\)。

14 个数的顺序是：

1. 全比对 \(R_i^{\mathrm{mean}}\)
2. 全比对 \(\Delta E_i\)
3. \(R_i^{\mathrm{mean}}\) 的样本标准差
4. \(\Delta E_i\) 的样本标准差
5. 全比对 \(R_{\mathrm{source}}\)
6. 它的样本标准差
7. 全比对 \(R_{\mathrm{other}}\)
8. 全比对 \(R_{\mathrm{contact}}\)
9. 全比对 \(R_{\mathrm{remote}}\)
10. 全比对 \(R_{\mathrm{unknown}}\)
11. 到 14. 第 7 到第 10 项各自的样本标准差

每个参数组单独满足 \(\Delta E_i=R_i^{\mathrm{mean}}+R_{\mathrm{source}}+R_{\mathrm{other}}\) 和 \(R_{\mathrm{other}}=R_{\mathrm{contact}}+R_{\mathrm{remote}}+R_{\mathrm{unknown}}\)。这不意味着标准差也能这样相加。全比对和 100 个重复都计算这两条残差的最大绝对值。掩码为真的 14 个数必须有限。

`energy_components.mutation_components` 只做互斥分组，不能表达同一伙伴上的 \(g_{ij}\) 和 \(1-g_{ij}\)。它可以在合成例子里核对叶瓣和，不能代替几何三项。\(\Delta E_i\) 同时用 `potts_energy.mutation_energy` 核对。

`记录.md` 第 185 行和第 454 行仍把 82、182 的坐标写成尚未取值。后来封口的规则是这两处不在坐标集。提取器按封口规则，不按那两句。`记录.md` 第 433 行写明能量恒等式的有限精度容差还没有取值；未取值时，既不把浮点残差当成失败，也不另选一个容差。因此正式提取可以报告最大残差，但不能在容差补上之前叫作恒等式已经通过。\(q_j\) 对状态求和的最大偏差同样报告，非有限就停止；什么样的偏差算「不是 1」，也还没有数值界。

这个模块不训练残差，不改 μFormer，不实现 OOF，也不做消融。消融以后只在这 14 个数上改变掩码。

## 预计时间

下面是计时外推，不是正式提取。页缓存中的一次 `j_star.npy` 读取是 0.054 秒，文件 244028232 字节。对全部 263 个位点、21 个状态做一遍七个分量是 0.125 秒。101 个参数组的计算约 13 秒。现有 `_background` 的一列约 0.017 秒，263 列约 4 秒，只做一次。100 个 `j_star.npy` 合计约 24.4 GB；按 0.5–2 GB/s 估计，读盘约 12–50 秒。正式提取的量级是一两分钟，不是一次 Bootstrap。即使如此，本轮也不启动。

## 本轮文件

修改 `dcr/contact_validation.py`、`dcr/test_contact_validation.py`。新增 `dcr/contact_validation_audit.json` 和本设计。不覆盖 `dcr/contact_validation_summary.json`。

将来审查通过后才新增 `dcr/potts_features.py` 和 `dcr/test_potts_features.py`。特征表和验收数字写到仓库外或被忽略的目录，不提交 `npy`。

## 尚未做

系统发育零假设和 MI 对照没有设计，也没有运行。14 维提取代码没有写。OOF、残差网络和 μFormer 都没有动。
