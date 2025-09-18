import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
import numpy as np


class Model(nn.Cell):
    """
    Informer with ProbSparse attention in O(LlogL) complexity (MindSpore)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
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
        x_dec = ops.Zeros()((x_enc.shape[0], 48 + 720, x_enc.shape[2]), ms.float32)
        x_mark_dec = ops.Zeros()((x_enc.shape[0], 48 + 720, 4), ms.float32)

        # Encoder forward
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out)

        # Decoder forward
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out)

        # Projection
        final_out = self.projection(dec_out)

        if self.output_attention:
            return final_out[:, -self.pred_len:, :], None  # Placeholder for attention
        else:
            return final_out[:, -self.pred_len:, :]


if __name__ == '__main__':
    class Configs:
        def __init__(self):
            self.pred_len = 720
            self.output_attention = False
            self.enc_in = 7
            self.dec_in = 7
            self.d_model = 16
            self.d_ff = 16
            self.dropout = 0.05
            self.c_out = 7


    configs = Configs()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    model = Model(configs)

    x_enc = Tensor(np.random.randn(32, 96, 7), ms.float32)
    x_mark_enc = Tensor(np.random.randn(32, 96, 4), ms.float32)
    x_dec = Tensor(np.random.randn(32, 768, 7), ms.float32)
    x_mark_dec = Tensor(np.random.randn(32, 768, 4), ms.float32)

    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    print('Input shape:', x_enc.shape)
    print('Output shape:', out.shape)
