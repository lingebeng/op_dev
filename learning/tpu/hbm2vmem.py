# ==========================================
# JAX Pallas TPU: 手动内存调度 (HBM -> VMEM)
# ==========================================

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


# ------------------------------------------
# 1. 定义底层 Kernel (在 TPU 物理核心上运行)
# ------------------------------------------
def hbm_vmem_kernel(x_hbm_ref, out_vmem_ref, scratch_vmem_ref):
    """
    参数解析：
    - x_hbm_ref: 远在 HBM 的输入张量引用 (只能读写，不能直接计算)
    - out_vmem_ref: 位于 VMEM 的输出张量引用 (自动分块好的目的地)
    - scratch_vmem_ref: 位于 VMEM 的暂存区引用 (我们手动申请的“工作台”)
    """

    # [物理动作 1]：调用 DMA 引擎，从 HBM 中精准截取第 0 行 (形状 1x128)
    # 并将其同步拷贝到我们位于 VMEM 的暂存区。
    # sync_copy 会阻塞计算单元，直到搬运完成。
    pltpu.sync_copy(x_hbm_ref.at[0:1], scratch_vmem_ref)

    # [物理动作 2]：数据已在极速的 VMEM 中。
    # 唤醒向量单元 (VPU)，对这 128 个元素并行加 1，并直接写回输出地址。
    out_vmem_ref[...] = scratch_vmem_ref[...] + 1


# ------------------------------------------
# 2. 宿主机主程序 (在 CPU 上运行，调度 TPU)
# ------------------------------------------
def main():
    # 初始化一个随机数生成器
    key = jax.random.key(0)

    # 在 HBM 中生成一个 8x128 的浮点数矩阵
    print("正在 HBM 中生成输入张量 x (8x128)...")
    x = jax.random.uniform(key, (8, 128), jnp.float32)

    # 调用 Pallas Kernel
    print("正在调用 Pallas Kernel 执行硬件级计算...")
    out = pl.pallas_call(
        hbm_vmem_kernel,
        # in_specs: 告诉编译器 x 留在 HBM (pl.ANY)，不要自动搬运
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        # out_shape: 声明我们的最终输出是一个 1x128 的矩阵
        out_shape=jax.ShapeDtypeStruct((1, 128), jnp.float32),
        # scratch_shapes: 手动向 TPU 申请一块位于 VMEM 的持久化暂存区
        scratch_shapes=(pltpu.VMEM(shape=(1, 128), dtype=jnp.float32),),
    )(x)

    # ------------------------------------------
    # 3. 结果验证
    # ------------------------------------------
    # 用高级 JAX/NumPy 语法计算标准答案：取 x 的第 0 行并加 1
    expected_out = x[0:1] + 1

    # 对比底层 Kernel 的结果与标准答案
    np.testing.assert_allclose(out, expected_out)
    print("✅ 验证成功！手动 DMA 拷贝与 VPU 计算结果完全一致。")


# 执行主程序
if __name__ == "__main__":
    main()
