import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np

class Model(nn.Cell):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 获取参数
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, has_bias=False, pad_mode="pad")

        # 全连接层
        self.linear = nn.Dense(self.seg_num_x, self.seg_num_y, has_bias=False)

    def construct(self, x):
        batch_size = x.shape[0]

        # 标准化并调整维度：b, s, c -> b, c, s
        #print(f"x shape: {x.shape}")  # 输出 x 的维度
        seq_mean = ops.ReduceMean(keep_dims=True)(x, axis=1)
        #print(f"x shape: {x.shape}")  # 输出 x 的维度
        x = ops.Transpose()(x - seq_mean, (0, 2, 1))

        # 一维卷积聚合
        x = self.conv1d(x.reshape((-1, 1, self.seq_len))).reshape((-1, self.enc_in, self.seq_len)) + x

        # 下采样：b, c, s -> bc, n, w -> bc, w, n
        x = x.reshape((-1, self.seg_num_x, self.period_len))
        x = ops.Transpose()(x, (0, 2, 1))

        # 稀疏预测
        y = self.linear(x)  # bc, w, m

        # 上采样：bc, w, m -> bc, m, w -> b, c, s
        y = ops.Transpose()(y, (0, 2, 1)).reshape((batch_size, self.enc_in, self.pred_len))

        # 调整维度并去标准化
        y = ops.Transpose()(y, (0, 2, 1)) + seq_mean

        return y