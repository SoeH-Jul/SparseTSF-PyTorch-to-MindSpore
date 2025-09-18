import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
import numpy as np

# 设置运行环境
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Model(nn.Cell):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = nn.Dense(configs.enc_in, configs.d_model)
        self.dec_embedding = nn.Dense(configs.dec_in, configs.d_model)

        # Encoder
        self.encoder = nn.SequentialCell([
            nn.Dense(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        ])

        # Decoder
        self.decoder = nn.SequentialCell([
            nn.Dense(configs.d_model, configs.d_ff),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        ])

        self.projection = nn.Dense(configs.d_model, configs.c_out)

    def construct(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_mark_enc = ops.Zeros()((x_enc.shape[0], x_enc.shape[1], 4), ms.float32)
        x_dec = ops.Zeros()((x_enc.shape[0], self.label_len + self.pred_len, x_enc.shape[2]), ms.float32)
        x_mark_dec = ops.Zeros()((x_enc.shape[0], self.label_len + self.pred_len, 4), ms.float32)

        # Encoder embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)

        # Decoder embedding
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out)

        # Projection
        final_out = self.projection(dec_out)

        if self.output_attention:
            return final_out[:, -self.pred_len:, :], None  # Attention not implemented
        else:
            return final_out[:, -self.pred_len:, :]


if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.seq_len = 336
            self.label_len = 48
            self.pred_len = 720
            self.output_attention = True
            self.enc_in = 7
            self.dec_in = 7
            self.d_model = 16
            self.embed = 'timeF'
            self.dropout = 0.05
            self.factor = 1
            self.n_heads = 8
            self.d_ff = 16
            self.e_layers = 2
            self.d_layers = 1
            self.moving_avg = [25]
            self.c_out = 7
            self.activation = 'gelu'
            self.wavelet = 0


    configs = Configs()
    model = Model(configs)

    enc = Tensor(np.random.randn(32, configs.seq_len, 7), ms.float32)
    enc_mark = Tensor(np.random.randn(32, configs.seq_len, 4), ms.float32)

    dec = Tensor(np.random.randn(32, configs.label_len + configs.pred_len, 7), ms.float32)
    dec_mark = Tensor(np.random.randn(32, configs.label_len + configs.pred_len, 4), ms.float32)

    out = model(enc, enc_mark, dec, dec_mark)
    print('Input shape:', enc.shape)
    print('Output shape:', out.shape)

    def count_parameters(model):
        return sum([np.prod(param.shape) for param in model.trainable_params()])


    print('Model size (in parameters):', count_parameters(model) / (1024 * 1024))
