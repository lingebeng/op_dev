# reduce_sum1.py vs reduce_sum2.py 分析

两个文件都用 JAX Pallas 实现了沿 axis=0 的 reduce sum，但策略完全不同。

## reduce_sum1.py — "一口吞"策略

- **Grid 是 2D**：`(M // block_M, N // block_N)`，只覆盖输出维度
- **输入 BlockSpec** 的 `block_shape=(B, block_M, block_N)`，把整个 reduction 轴（B=8）一次性加载进 SRAM
- **Kernel 内部**：一次 `jnp.sum(x_ref[...], axis=0)` 直接完成归约
- 每个 grid cell 独立完成计算，**不需要跨 grid 累加**

## reduce_sum2.py — "逐片累加"策略

- **Grid 是 3D**：`(M // blk_M, N // blk_N, reduction_size)`，reduction 轴被拆进 grid 的第 3 维
- **输入 BlockSpec** 用 `None` 表示 reduction 轴每次只取 1 片（自动 squeeze）
- **Kernel 内部**：
  - `k == 0` 时先清零 `o_ref`
  - 然后 `o_ref[...] += x_ref[...]` 逐步累加
- 同一个输出块被**多次 grid 迭代写入**

## 核心区别总结

| | reduce_sum1 | reduce_sum2 |
|---|---|---|
| Grid 维度 | 2D | 3D（reduction 轴在 grid 中） |
| SRAM 占用 | 一次加载整个 reduction 轴 | 每次只加载 1 片 |
| Kernel 逻辑 | 单次 `jnp.sum` | 初始化 + 累加 |
| 可扩展性 | reduction 轴必须能放进 SRAM | reduction 轴可以任意大 |

**简单说**：sum1 靠 SRAM 容量"吃下"整个 reduction 轴一步到位；sum2 把 reduction 轴拆成循环，每次只处理一片，牺牲一些 kernel 调度开销换取对大 reduction 轴的支持。当 B 很大、SRAM 放不下时，只能用 sum2 的方式。
