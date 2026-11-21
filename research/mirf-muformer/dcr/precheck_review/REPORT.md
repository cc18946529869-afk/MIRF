# 预检只读复算

2026-11-21。对象是已经保存的 `research/mirf-muformer/dcr/precheck/`。没有重新生成索引，没有改种子，没有改数学假设和落实方案，没有训练，没有跑 Bootstrap。

审查进程不能启动命令。指定解释器和 `sha256sum` 都在执行前被拒绝，该交回写「这一步未完成」，不写 GO。下面的数来自已经跑通的终端，不是那次被拒绝的进程。

## 命令

```text
env -u PYTHONPATH /home/cc233/miniforge3/envs/nsf310/bin/python research/mirf-muformer/dcr/precheck_review/recompute_readonly.py
```

原始输出：`recompute_stdout.txt`。

```text
sha256sum -c SHA256SUMS
```

工作目录：`research/mirf-muformer/dcr/precheck`。原始输出：`sha256sum_check.txt`，27 行都是 OK。

## 结果

| 检查 | 结果 |
| --- | --- |
| 解释器 | `/home/cc233/miniforge3/envs/nsf310/bin/python` |
| NumPy | 1.23.5，`.../site-packages/numpy/__init__.py` |
| sklearn | 1.7.2，`.../site-packages/sklearn/__init__.py` |
| SHA256SUMS | 27 个条目全部相符，`sha256_mismatch_count 0` |
| n=1000 验证折 | 200 / 200 / 200 / 200 / 200，两两互斥，并成 `0..999` |
| n=1000 折训练行 | 每折 800，且是该验证折的补集 |
| 全量验证折 | 700 / 700 / 699 / 699 / 699，两两互斥，并成 `0..3496` |
| 全量折训练行 | 2797 / 2797 / 2798 / 2798 / 2798，且是补集 |
| 与 `KFold(5, shuffle=True, random_state=20261121)` | 两种规模、五折的训练集合和验证集合都相等 |
| n=1000 内部划分 | 900 与 100，不交，并成 `0..999` |
| 全量内部划分 | 3147 与 350，不交，并成 `0..3496` |
| 与 `train_test_split(..., test_size=0.1, random_state=20261122, stratify=None)` | 集合相等，保存的行序也相等 |
| 六份 TSV 文件内重复序列 | 都是 0 |
| train/valid/test 序列交集 | 两种规模都是 0、0、0 |
| n=1000 训练序列不在全量 train 中的条数 | 0 |
| 两套 valid、test | 逐行相同 |
| 套用全量折号后折号改变的行数 | 803 |
| 十个外层整数 | 互异，与 42、20261121、20261122、20261123、2847319051 无碰撞 |

## 结论

预检划分 GO。`dcr/precheck/` 里保存的索引就是封口清单的划分。

代码与训练 NO-GO。这个 GO 不恢复最佳权重，不写残差分支，不开始训练。用户再说一次可以，才进入实现。
