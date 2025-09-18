
import mindspore as ms
from mindspore import ops
import numpy as np

class TriangularCausalMask:
    def __init__(self, B, L):
        self.mask = ops.triu(ms.Tensor(np.ones((B, 1, L, L)), ms.float32), diagonal=1)

    def get_mask(self):
        return self.mask
