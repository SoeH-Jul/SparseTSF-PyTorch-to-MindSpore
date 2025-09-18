import numpy as np
import mindspore as ms
from mindspore import Model

# 这里删除不必要的导入
# from sympy.physics.units import ms


class Configs:
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 48
        self.enc_in = 7

configs = Configs()

# 假设模型是自定义的一个继承了 nn.Cell 的类
# 需要在代码中创建一个模型对象，示例如下：
class SimpleModel(ms.nn.Cell):
    def __init__(self, configs):
        super(SimpleModel, self).__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        # 示例: 简单的线性层，按需修改
        self.dense = ms.nn.Dense(self.enc_in, 64)

    def construct(self, x_enc):
        return self.dense(x_enc)

# 创建模型实例
model = SimpleModel(configs)

# 创建输入数据
x_enc = ms.Tensor(np.random.randn(32, configs.seq_len, configs.enc_in), ms.float32)

# 调用模型进行前向传播
y = model.construct(x_enc)

print("Input shape:", x_enc.shape)
print("Output shape:", y.shape)
