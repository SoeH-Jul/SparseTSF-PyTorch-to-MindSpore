__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding',
           'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding']

import mindspore
from mindspore import nn, ops, Tensor
import math


class Transpose(nn.Cell):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def construct(self, x):
        return ops.transpose(x, self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


# decomposition

class moving_avg(nn.Cell):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, pad_mode='valid')

    def construct(self, x):
        # Padding on both ends of time series
        front = ops.tile(x[:, 0:1, :], (1, (self.kernel_size - 1) // 2, 1))
        end = ops.tile(x[:, -1:, :], (1, (self.kernel_size - 1) // 2, 1))
        x = ops.concat([front, x, end], axis=1)
        x = self.avg(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        return x


class series_decomp(nn.Cell):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def construct(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    pe = ops.zeros((q_len, d_model), mindspore.float32)
    position = ops.arange(0, q_len).view(-1, 1)
    div_term = ops.exp(ops.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = ops.sin(position * div_term)
    pe[:, 1::2] = ops.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    for _ in range(100):
        cpe = 2 * (ops.linspace(0, 1, q_len).view(-1, 1) ** x) * (ops.linspace(0, 1, d_model).view(1, -1) ** x) - 1
        if abs(cpe.mean().asnumpy()) <= eps:
            break
        elif cpe.mean().asnumpy() > eps:
            x += .001
        else:
            x -= .001
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (ops.linspace(0, 1, q_len).view(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe is None:
        W_pos = Tensor(mindspore.numpy.random.uniform(-0.02, 0.02, (q_len, d_model)), mindspore.float32)
        learn_pe = False
    elif pe == 'zero':
        W_pos = Tensor(mindspore.numpy.random.uniform(-0.02, 0.02, (q_len, 1)), mindspore.float32)
    elif pe == 'zeros':
        W_pos = Tensor(mindspore.numpy.random.uniform(-0.02, 0.02, (q_len, d_model)), mindspore.float32)
    elif pe in ['normal', 'gauss']:
        W_pos = Tensor(mindspore.numpy.random.normal(0.0, 0.1, (q_len, 1)), mindspore.float32)
    elif pe == 'uniform':
        W_pos = Tensor(mindspore.numpy.random.uniform(0.0, 0.1, (q_len, 1)), mindspore.float32)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid positional encoding type.")
    return mindspore.Parameter(W_pos, requires_grad=learn_pe)
