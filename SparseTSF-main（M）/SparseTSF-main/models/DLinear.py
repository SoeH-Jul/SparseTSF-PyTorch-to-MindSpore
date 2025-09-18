import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class MovingAvg(nn.Cell):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def construct(self, x):
        # Padding on both ends of time series
        front = ops.Tile()(x[:, 0:1, :], (1, (self.kernel_size - 1) // 2, 1))
        end = ops.Tile()(x[:, -1:, :], (1, (self.kernel_size - 1) // 2, 1))
        x = ops.Concat(1)((front, x, end))
        x = self.avg(x.swapaxes(1, 2))
        x = x.swapaxes(1, 2)
        return x


class SeriesDecomp(nn.Cell):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def construct(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Cell):
    """
    Decomposition-Linear
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = SeriesDecomp(kernel_size)
        self.individual = configs.individual
        self.enc_in = configs.enc_in
        self.period_len = 24

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.Linear_Seasonal = nn.Dense(self.seq_len, self.pred_len, has_bias=False)
        self.Linear_Trend = nn.Dense(self.seq_len, self.pred_len, has_bias=False)

    def construct(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init = ops.Transpose()(seasonal_init, (0, 2, 1))
        trend_init = ops.Transpose()(trend_init, (0, 2, 1))

        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return ops.Transpose()(x, (0, 2, 1))  # to [Batch, Output length, Channel]
