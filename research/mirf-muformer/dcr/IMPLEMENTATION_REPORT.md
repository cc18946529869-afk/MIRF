# DCR 残差分支实现报告

2026-11-22。分支 `research/dcr-residual-implementation-20261121`。正式十种子训练没有启动，也没有放行。没有真实适应度结果，没有对真实实验输出计算融合指标，也没有写成已经超过 baseline。

## 参数传递

封口七项从 `sealed_config.SEALED` 进入 `outer_train.train_outer_models`、`oof.run_seed_protocol` 和 `residual.fit_predictor`。三处执行前都调用 `require`。配置和封口取值不一致就拒绝。

- OOF 的 `KFold` 种子是 `20261121`，内部 10% 的种子是 `20261122`。预检只读 `dcr/precheck/`，核对目录名和文件修改时间，不重新生成索引。
- \(E_s\) 是该外层种子自己的 baseline 最佳轮。五个折模型共用这一轮。来源必须是 `this_outer_seed_baseline`。种子 42 的 229 不是默认轮数，也没有写进这三处执行文件。折请求若把 229 当成另一轮 baseline 的 \(E_s\)，或者 `select_on_held_out_fold` 为真，就在调用 `train` 之前拒绝。
- 残差执行层使用并断言 AdamW、学习率 \(1\times10^{-3}\)、weight decay \(1\times10^{-4}\)、批大小 64、最多 300 轮、调度器为 `None`、早停指标 MAE、每轮评估、patience 30。调用方不能改成 patience 1、batch size 250 或 max epochs 2。
- 批大小 64 不进入 μFormer 的请求，也不传给 `trainer.train`。比较配置的批大小 16、两组学习率、weight decay 0、hidden 256、dropout 0.2、siamese、warmup 10 不在这一步的请求里。
- 读出仍是 \(d_s=64,d_l=32,d_r=32\)、拼接 160、64 维 GELU、dropout 0.1、输出零初始化。执行时断言这些值。
- 特征表只读已经生成的 `features/soft_weight_coords_audit.json`。策略是 `soft_weight_keeps_82_and_182`，参考位置 82 与 182 在坐标集内，审计记录没有把错配补集当作坐标集。这一检查不调用 `in_coord = ~mismatched`，不写 v0.1，不重算特征表。

没有改 Potts 分解、14 维、残差网络结构、七项取值、μFormer 的评分损失和优化器、评分隔离。没有重拟合，没有重跑 Bootstrap，没有新增 \(R_J\) 前置。`run_ten_outer_seeds` 仍直接拒绝。

## 合成测试

参数传递和这一节分开。正式配置没有被缩短。

- 三臂每一轮选轮顺序和每一轮重训顺序，在 12 行小型合成数据上由真实的 `fit_predictor` 跑出。超参数仍是封口值。三臂在共同轮次上的顺序相同，每一臂的全部选轮顺序和全部重训顺序都能由该阶段自己的随机数流重放。
- n=1000 的协议检查和 `rehearsal.rehearse` 用记录调用的替身。替身记下的是封口配置：AdamW、学习率 \(1\times10^{-3}\)、weight decay \(1\times10^{-4}\)、批大小 64、最多 300 轮、无调度器、MAE、patience 30，以及预检里的 900 与 100 行。不再把 patience 1、batch size 250、max epochs 2 传进执行。
- 预检目录的修改时间在读取后不变。软权重审计、软权重表和 v0.1 的修改时间在核验后不变。

解释器 `/home/cc233/miniforge3/envs/nsf310/bin/python`。工作目录 `research/mirf-muformer/dcr`。

```text
env -u PYTHONPATH /home/cc233/miniforge3/envs/nsf310/bin/python -m unittest test_best_weights.py test_residual_protocol.py test_outer_train.py test_rehearsal.py test_metrics.py
```

Ran 30 tests in 2.493s，OK。

## 正式训练

正式十种子训练仍是 NO-GO。`run_ten_outer_seeds` 在构造训练器之前拒绝。没有真实适应度结果。用户另说可以，才进入正式训练。

## 分支上已经有的状态

这些是这一步之前已经写进代码、这一步没有改掉的事实。

- 验证评分显式使用外层验证集。测试评分单独调用，并且只在 \(\eta\) 锁定之后计算融合指标。`choose_eta` 不接收测试集标签。
- 三条残差臂共用同一份初始参数和同一套样本顺序。选轮和最终重训使用分开的随机数流。
- 软权重特征表是 `dcr/features/blat_ecolx_stiffler_2015_soft_weight_coords.tsv`，表头加 4996 行数据。`blat_ecolx_stiffler_2015_v0.1.tsv` 未改。
- `Muformer.forward` 的评分、MAE 损失、Adam 和原来的学习率调度没有改。`torch.load` 仍是注释。\(R_J=0.4581354189599478\)，`action` 仍是 `parameter_unstable`。
