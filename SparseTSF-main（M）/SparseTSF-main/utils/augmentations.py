
import mindspore as ms
from mindspore import ops
import numpy as np

def freq_mask(x, rate=0.5, dim=1):
    x_f = ops.rfft(x, dim)
    mask = ms.Tensor(np.random.rand(*x_f.shape)) < rate
    x_f.real = ops.select(mask, ms.Tensor(0, ms.float32), x_f.real)
    x_f.imag = ops.select(mask, ms.Tensor(0, ms.float32), x_f.imag)
    return ops.irfft(x_f, dim)
