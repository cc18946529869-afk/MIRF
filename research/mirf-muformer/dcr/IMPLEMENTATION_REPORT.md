# DCR 残差分支实现报告

2026-11-21。分支 `research/dcr-residual-implementation-20261121`，起点是预检审查提交 `adb88aa`。正式训练没有启动。没有计算十个种子的融合指标，也没有写成已经超过 baseline。

## 实际修改的文件

- `Mu-Protein/mu-former/src/trainer.py`：记下最佳轮，训练循环结束后把 `best_model_state` 装回同一个 `model.module`。
- `research/mirf-muformer/dcr/lobes.py`：成熟位点 24–286，信号肽不进叶。
- `research/mirf-muformer/dcr/geometry.py`：软权重坐标集保留参考位置 82、182 的 Cβ，不使用 `in_coord = ~mismatched`。
- `research/mirf-muformer/dcr/frozen_pmlm.py`：独立预训练编码器，`eval` 且不求梯度，读最后一层隐状态。
- `research/mirf-muformer/dcr/residual.py`：14 维分开进入，28 维拼接，宽度 64/32/32，64 维 GELU，dropout 0.1，输出层零初始化。
- `research/mirf-muformer/dcr/predict.py`：在 `forward` 外计算 \(s_{\mu F}+\eta r\)。
- `research/mirf-muformer/dcr/oof.py`：只读 `dcr/precheck/`，五折、内部 10%、三条臂。`mode` 不是 `synthetic` 时拒绝执行。
- `research/mirf-muformer/dcr/bootstrap_inventory.py` 及它写出的清单、核验报告、提交文件列表。
- `research/mirf-muformer/dcr/test_best_weights.py`、`test_residual_protocol.py`。

`Muformer.forward` 的评分、MAE 损失、Adam 和原来的学习率调度没有改。`torch.load` 仍是注释。没有重拟合 Potts，没有重跑 Bootstrap，没有把 `.npy` 放进提交清单。

## 模块逻辑

最佳轮数只在 `is_best` 为真时写入 `BestEpochWeights`。随后不是最佳的评估返回 None，不会把已经记下的轮数清掉。`_evaluate_and_save_best` 在最佳时用 `copy.deepcopy` 保存 `model.module.state_dict()`。循环结束后 `restore_best_model_state` 把这份副本装回同一个模块。

残差预测器的 `forward` 只返回 \(r\)。融合函数是 `predict.fuse`。\(\eta\) 在 \(\{0,0.05,\ldots,1\}\) 上选择，顺序是验证集 Spearman、RMSE、更小的 \(\eta\)。总和对照只保留 `delta_e` 和 `std_delta_e` 的掩码。无 Potts 对照把 14 个槽全部掩码，位点和叶瓣投影仍在。三条臂各自调用 `fit_predictor`，各自选 \(\eta\)。同一外层种子传入的 \(s_{\mu F}\) 和 \(E_s\) 不在臂之间改写。

OOF 标签来自调用方提供的 `fit_fold`。五个折都用同一个 \(E_s\)，不用残差的 patience 30。残差轮数只在内部验证 MAE 创新低时更新。选出轮数后重新初始化，再在全部 OOF 行上训练该轮数。

## 测试

解释器 `/home/cc233/miniforge3/envs/nsf310/bin/python`。工作目录 `research/mirf-muformer/dcr`。

```text
python -m unittest test_best_weights.py test_residual_protocol.py -v
```

10 个测试，0.752 秒，全部通过。覆盖最佳轮不被非最佳评估清掉、同一模块恢复最佳权重、`torch.load` 仍被注释、评分公式未改、预检索引哈希未改、82 与 182 留在软权重集、冻结编码器、宽度和掩码、融合在 `forward` 外、正式模式被拒绝、合成 n=1000 上三臂共用一份分数和 \(E_s=5\)。

`python bootstrap_inventory.py` 退出码 0。`passed_check` 为真。100 个 replicate 都在。\(R_J=0.4581354189599478\)，`action` 仍是 `parameter_unstable`，`passed` 仍为假。`h_hat.npy` 与 `j_star.npy` 的文件 SHA256 与 `potts_features.py` 里锁定的规范哈希一致。每个 replicate 的 263 个 `site_*.npz` 只点数，没有算 SHA256。

## 数学恒等式和特征顺序

14 列顺序是 `potts_features.FEATURE_NAMES`：`r_mean`、`delta_e`、`std_r_mean`、`std_delta_e`、`r_source`、`std_r_source`、`r_other`、`r_contact`、`r_remote`、`r_unknown`，以及后四项的标准差。输入是 \([x\odot m; m]\)，宽度 28。拼接宽度是 \(64+32+32+32=160\)。能量恒等式仍由已有的 `identity_audit` 承担。这一步没有用新坐标集重写 4996 行特征表，残差代码也不读旧的 `dcr/features/blat_ecolx_stiffler_2015_v0.1.tsv`。

## 数据隔离

`oof.load_precheck` 重算 `SHA256SUMS` 的 27 个条目，并检查两套索引各自是划分、互不混用。n=1000 与全量的文件名分开。测试断言预检文件的修改时间没有变。正式模式直接拒绝，所以这条路径不会启动十个种子的真实训练。

## Bootstrap 身份

只读路径是主检出里的 `research/mirf-muformer/dcr/bootstrap/`。工作树里没有复制这些数组。清单记录路径、字节数、SHA256、形状和 dtype。这不是系统发育校正已经完成。

## 尚未落到可执行训练上的句子

- 十个配对种子上的 Spearman 三句、Pearson、NDCG@10%、Top-10% Recall、RMSE 和 10000 次配对 Bootstrap 没有实现成正式指标。
- 「每一折用其余四折从头训练一个原始 μFormer」只留了 `fit_fold` 接口。测试里它返回合成分数，没有调用 `Trainer.train`。
- `load_independent_pretrained` 没有对 `pretrained/pmlm_650m.pt` 执行。
- 用保留 82、182 的坐标集重算特征表还没有做。

## 正式训练的资源和时间

这一步没有重跑计时。已有 n=1000、种子 42 的日志第 1 轮是 172.1 秒。正式日程是两个规模、十个外层种子，每个种子一个完整 μFormer 再加五个折模型，折模型轮数是该种子的 \(E_s\)。残差预测器相对 μFormer 很小。按这一条已有日志粗算，仅 n=1000 的十个完整模型和五十个折模型就要数百 GPU 小时，全量训练更长。日志没有写明卡数，这里不把 172.1 秒写成全量或后续轮次的测量值。正式训练仍要另一次可以。
