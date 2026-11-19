# 划分身份与数值容差审查

这是正式提取前的核对，不是特征表，也不是已经写入 `记录.md` 的协议。三条结论仍分开：\(J^\star\) 对 1BTL 近距离残基的富集已有支持；系统发育零假设和 MI 对照未做；突变效应预测未做。本轮没有写 `potts_features.py`，没有重跑 Bootstrap，没有改 `trainer.py`，也没有覆盖 `contact_validation_summary.json`。

## 原始划分

服务器上找到了 v0.1 划分，没有重新抽样，也没有用 v1.3 的 CV folds。

可用文件在

`/data/cc233_data/work_dir/MIRF/proteingym_v0.1_split/subs/`

另一份字节相同的副本在

`/data/cc233_data/work_dir/Muformer_DPFunc/proteingym_v0.1_split/subs/`

baseline 实际打开的是

`/data/cc233_data/work_dir/Muformer_DPFunc/new-Mu-Protein-main/data/dms/BLAT_ECOLX_Stiffler_2015/`

这三处的 train、valid、test 字节相同。`n1000` 的 test 和 valid 是指向上述 baseline 文件的符号链接。

| 集合 | 行数 | SHA256 |
| --- | ---: | --- |
| train | 3497 | `615e47bd09ca1a413e77449dbf286036d227d588cfc12034feff70c7a9ffbe67` |
| valid | 500 | `803af89ec564da0f84b857e3c6efbb96a9718e4db45d7ab853294b87f7406fc6` |
| test | 999 | `66e8ce3eec1e603e435ff54111bc95804b64104a4f5da90fb1cf45a569f9854d` |

三份文件的列是 `sequence`、`score`、`score_bin`，没有 `mutation` 列。baseline 的 `data.py` 在已有 `sequence` 列时直接使用该列，不从 `mutation` 生成序列。因此「`mutation` 字段完整」这一句和文件本身不符。能核对的样本身份是完整序列。

以 286 位野生型逐行比较后，4996 行都恰好是一个氨基酸替换。由此推出的 `WiM` 在三个集合内部唯一，集合之间没有交集，合并后也是 4996 个不同字符串。这个 `WiM` 是审查时按序列差推出来的，不是文件里的一列。按文件顺序、每行一个字符串、UTF-8、最后不额外加空行，三个集合的 SHA256 是：

| 集合 | 推出的 WiM SHA256 |
| --- | --- |
| train | `778d9703a65c92a251fbd9c3efb823960fb509b4a90a7ed66ed0c7e0eb4807c0` |
| valid | `86ce867b283ae737f64ca381dc202bca9294f440aa378c3450baff3e02f36f33` |
| test | `e864fcb2598c463243044e8f5b279d1d7381664b42b9189beaf56195bfea030e` |

`baseline/n1000/prediction.tsv`、`baseline/full/prediction.tsv` 和 `baseline/n300/prediction.tsv` 都是 999 行，`sequence` 与 `score_bin` 和 test 文件逐行相同。保存的 `score` 印到了小数点后 8 位，与 test 原值的最大绝对差是 \(5.0\times 10^{-9}\)。

`n1000` 训练集是

`/data/cc233_data/work_dir/Muformer_DPFunc/new-Mu-Protein-main/data/dms/BLAT_ECOLX_Stiffler_2015_n1000/BLAT_ECOLX_Stiffler_2015_n1000_train.tsv`

SHA256 是 `6099139df983b7c5efd232013e8d8242674d310c8e509562283584b8c27c46b1`。1000 行的 `sequence`、`score`、`score_bin` 都原样出现在 3497 行训练集里，而且保持训练集中的先后顺序。它不是前 1000 行，下标从 1 到 3491，中间有空位。推出的 WiM SHA256 是 `d0387620d4954b1da1ec38fd4c44ea5fee96d40bec9bd529c3dc671b1a8a3dc2`。`n1000` 的日志写的就是这个训练文件，测试文件则是上面的 test。valid 没有单独的预测表；`n1000` 的 valid 链接指向同一份 500 行文件。

`/data/cc233_data/work_dir/protein expression level dataset/proteingym_v0.1_split (2)/release/proteingym_v0.1_split/subs/` 里同名的三个文件大小是 0，不能用。

## 容差

候选标准没有写入正式协议。下面是用冻结公式和已保存参数量到的偏差，用来看 \(10^{-12}\) 会不会把正确的 float64 结果判失败。

背景 \(q\) 用 `solver_v2._background`，输入是 `mature_codes.npy`、`neighbor_counts.tsv` 的 \(1/m_i\)，以及 `OFFICIAL_M_EFF = 47899.73496814515`。结果形状是 \(263\times 21\)，`float64`，全部有限，最小值是 \(9.94\times 10^{-7}\)，没有负数。263 个位点里 \(\lvert\sum_b q_j(b)-1\rvert\) 的最大值是 \(6.66\times 10^{-16}\)。正式重复 1 到 100 已保存的 `q.npy` 都是 \(263\times 21\) 的 `float64`，有限且非负，同一偏差的最大值是 \(4.44\times 10^{-16}\)。

能量恒等式在全比对的 `h_hat.npy`、`j_star.npy` 上，对 263 个位点的全部 21 个目标状态各算一次。\(\Delta E\) 每次都调用 `potts_energy.mutation_energy`，不拿分解式自己减自己。最大 \(\epsilon_E\) 是 \(5.04\times 10^{-16}\)，最大 \(\epsilon_G\) 是 \(5.98\times 10^{-16}\)。5523 次里没有一次超过 \(10^{-12}\)。重复 1 的状态 0、10、20 上，最大 \(\epsilon_E\) 是 \(4.10\times 10^{-16}\)。100 个重复的 \(J^\star\) 没有逐组重算；正式提取时仍要保存每一组的最大原始残差和归一化残差。

\(10^{-12}\) 比这些实测大约宽三个数量级。按现在的求和长度，它不会误伤冻结的 `_background` 和这条分解。它也比漏掉一个普通大小的伙伴所产生的残差紧。分母取 1 和各绝对项之和的较大者，所以绝对误差若远小于分量本身，归一化之后会更小；这次观测到的原始残差是 \(10^{-16}\) 量级。

两条恒等式对 \(q\) 不敏感，因为 \(\mu_{ij}\) 和 \(u_{ij}\) 相加后消掉。\(q\) 的 \(10^{-12}\) 必须单独保留。恒等式通过只说明同一组数在 float64 里重新加总仍然闭合，不代替叶瓣、距离和 \(g_{ij}\) 的定义检查。

审查结论：这两项候选与这次量到的 float64 偏差相符，确认之后可以当作提取验收界，并且之后不能为了让某一组通过而放宽。在确认写进协议之前，它们仍只是候选。

## 还没有做

14 维特征表没有生成。OOF、残差训练和系统发育对照都没有开始。正式训练仍然是 NO-GO。
