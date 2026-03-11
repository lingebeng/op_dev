# Torch / Triton 精度对比结果

测试脚本: `test.py`

GPU 环境:

- `torch 2.10.0+cu126`
- `NVIDIA H100 80GB HBM3`

测试参数:

- `--exp-size 65536`
- `--m 512 --n 512 --k 512`

## 运行方式

先确认当前 Python 环境和 GPU 可见:

```bash
./.venv/bin/python - <<'PY'
import torch
print('torch_version', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device_name', torch.cuda.get_device_name(0))
PY
```

单次运行一个 dtype:

```bash
./.venv/bin/python test.py --dtype fp16 --exp-size 65536 --m 512 --n 512 --k 512
./.venv/bin/python test.py --dtype bf16 --exp-size 65536 --m 512 --n 512 --k 512
./.venv/bin/python test.py --dtype fp32 --exp-size 65536 --m 512 --n 512 --k 512
```

我这次实际就是分别运行了上面三条命令。

## 结论

- `fp16` 下，`torch.exp` 和 `triton.exp` 结果几乎一致；`torch.mm` 和 Triton matmul 完全一致。
- `bf16` 下，`torch.exp` 和 `triton.exp` 也基本一致；`torch.mm` 和 Triton matmul 完全一致。
- `fp32` 下，`triton.exp` 的误差比 `torch.exp` 更大一些，但还在接近量级。
- `fp32` 下，当前 Triton matmul kernel 的精度明显差于 `torch.mm`，需要单独修正。

## 详细结果

### fp16

#### exp

```text
torch.exp(cuda) vs ref  max_abs=6.457603e+00  mean_abs=1.827687e-01  rmse=6.075262e-01  max_rel=2.395873e-03
triton.exp vs ref       max_abs=6.457603e+00  mean_abs=1.827687e-01  rmse=6.075262e-01  max_rel=2.395873e-03
torch vs triton         max_abs=6.103516e-05  mean_abs=9.313226e-09  rmse=6.529362e-07  max_rel=8.347245e-04
```

#### matmul

```text
torch.mm(cuda) vs ref   max_abs=1.390721e-02  mean_abs=1.961394e-03  rmse=2.506991e-03  max_rel=1.316512e+01
triton mm vs ref        max_abs=1.390721e-02  mean_abs=1.961394e-03  rmse=2.506991e-03  max_rel=1.316512e+01
torch vs triton         max_abs=0.000000e+00  mean_abs=0.000000e+00  rmse=0.000000e+00  max_rel=0.000000e+00
```

### bf16

#### exp

```text
torch.exp(cuda) vs ref  max_abs=5.067097e+01  mean_abs=1.456329e+00  rmse=4.869886e+00  max_rel=1.958495e-02
triton.exp vs ref       max_abs=5.067097e+01  mean_abs=1.456589e+00  rmse=4.869825e+00  max_rel=1.958495e-02
torch vs triton         max_abs=4.000000e+00  mean_abs=7.568359e-03  rmse=1.739926e-01  max_rel=4.273504e-03
```

#### matmul

```text
torch.mm(cuda) vs ref   max_abs=1.296244e-01  mean_abs=1.569774e-02  rmse=2.003025e-02  max_rel=1.434138e+02
triton mm vs ref        max_abs=1.296244e-01  mean_abs=1.569774e-02  rmse=2.003025e-02  max_rel=1.434138e+02
torch vs triton         max_abs=0.000000e+00  mean_abs=0.000000e+00  rmse=0.000000e+00  max_rel=0.000000e+00
```

### fp32

#### exp

```text
torch.exp(cuda) vs ref  max_abs=3.791476e-04  mean_abs=5.931854e-06  rmse=2.124086e-05  max_rel=1.365022e-07
triton.exp vs ref       max_abs=1.215018e-03  mean_abs=3.068216e-05  rmse=1.072137e-04  max_rel=4.564905e-07
torch vs triton         max_abs=1.220703e-03  mean_abs=3.035888e-05  rmse=1.087437e-04  max_rel=5.102497e-07
```

#### matmul

```text
torch.mm(cuda) vs ref   max_abs=1.228707e-05  mean_abs=1.195832e-06  rmse=1.559868e-06  max_rel=7.583312e-03
triton mm vs ref        max_abs=2.566185e-02  mean_abs=4.127785e-03  rmse=5.170138e-03  max_rel=1.738887e+01
torch vs triton         max_abs=2.565765e-02  mean_abs=4.127781e-03  rmse=5.170135e-03  max_rel=1.461507e+02
```
