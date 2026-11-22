# DCR 残差分支实现报告

2026-11-22。分支 `research/dcr-residual-implementation-20261121`。正式十种子训练没有启动，也没有放行。没有对真实实验输出计算融合指标，没有适应度提升的结果，也没有写成已经超过 baseline。

## 这一步

只修验证集和测试集的评分隔离。没有改数学分解，没有改七项参数，没有新增模型结构，没有改 `Muformer.forward`、损失函数和原有优化器，没有启用 `torch.load`。

恢复最佳权重之后，验证评分显式使用外层验证集。`execute_muformer_request` 把 `valid_loader` 和 `valid_df` 交给 `score_validation`。两者缺一就拒绝，不会退回 `self.test_loader` 和 `self.test_df`。

测试评分是另一次调用。`main.py` 在 `train` 返回后调用 `score_test(trainer.test_loader, trainer.test_df)`。合成日程里的测试评分在 `score_restored_test_split`。这两处都不传 `checkpoint_dir`，都不调用 `load_pretrained_model`，用的是训练结束时已经装回最佳权重的同一个模块。返回的测试分数不交给 `choose_eta`。

`choose_eta` 只收到验证集标签、该验证集上的 μFormer 分数和残差预测。入口是 `oof.select_eta`，参数是 `y_valid`、`s_valid` 和验证集上的残差。它没有测试集标签这个参数。

样本身份是 `sequence`。`require_same_sequences` 按顺序比较序列。长度相同但序列不同会拒绝；序列相同但顺序不同也会拒绝。

η 锁定之后，`fuse_test_after_eta` 才在测试集上计算融合指标。`mode` 不是 `synthetic` 时拒绝。返回值里 `formal_metrics_computed` 和 `formal_experiment` 都是假。这条路径不读真实实验输出。

`run_ten_outer_seeds` 仍直接拒绝。`train_outer_models` 和 `run_seed_protocol` 在 `mode` 不是 `synthetic` 时仍拒绝。

## 分支上已经有的状态

这些是后来的提交已经写进代码的事实，用来替换本报告早先写过的「10 个测试」「PMLM 尚未加载」「软权重特征表尚未重算」。

- 三条残差臂共用同一份初始参数和同一套样本顺序。选轮和最终重训使用分开的随机数流。
- 软权重特征表是 `dcr/features/blat_ecolx_stiffler_2015_soft_weight_coords.tsv`，表头加 4996 行数据。`soft_weight_coords_audit.json` 的 `passed` 为真，参考位置 82、182 留在软权重坐标集。`blat_ecolx_stiffler_2015_v0.1.tsv` 未改。
- `pmlm_650m.pt` 做了一次冻结推理核验。`features/pmlm_650m_frozen_inference.json` 的 `passed` 为真，`embed_dim` 是 1280，输出形状是 288×1×1280，`requires_grad` 为假，`prob_calls` 是 0，没有读 `fc_reduce`。这不是训练，也没有下载新权重。
- 折模型的训练行号进入 `Trainer.train` 实际迭代的 batch，尾批不丢。
- 评价指标在 `dcr/metrics.py`：Spearman 三句、Pearson、NDCG@10%、Top-10% Recall、RMSE，以及配对 Bootstrap。默认 10000 次。`formal_experiment` 固定为假。没有拿它去报真实十种子的提升。
- 端到端预演在 `rehearsal.py`，只用合成数据。

`Muformer.forward` 的评分、MAE 损失、Adam 和原来的学习率调度没有改。`torch.load` 仍是注释。没有重拟合 Potts，没有重跑 Bootstrap。\(R_J=0.4581354189599478\)，`action` 仍是 `parameter_unstable`。

## 测试

解释器 `/home/cc233/miniforge3/envs/nsf310/bin/python`。工作目录 `research/mirf-muformer/dcr`。

```text
env -u PYTHONPATH /home/cc233/miniforge3/envs/nsf310/bin/python -m unittest test_best_weights.py test_residual_protocol.py test_outer_train.py test_rehearsal.py test_metrics.py
```

Ran 25 tests in 2.965s，OK。

这次新增的四项：

- `test_validation_and_test_lengths_stay_separate_and_eta_uses_validation`：ok。验证集 4 行、测试集 6 行，η 只用验证集。
- `test_same_length_different_sequences_are_rejected`：ok。长度相同但 sequence 不同，以及顺序不同，都拒绝。
- `test_choose_eta_does_not_read_test_labels`：ok。
- `test_final_test_score_uses_the_restored_module`：ok。最终测试评分走恢复后的同一个模块。

## 正式训练

正式十种子训练仍是 NO-GO。`run_ten_outer_seeds` 在构造训练器之前拒绝。用户另说可以，才进入正式训练。
