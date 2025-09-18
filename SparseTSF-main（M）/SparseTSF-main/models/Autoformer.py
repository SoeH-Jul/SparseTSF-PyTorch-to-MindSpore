import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_pos_temp, DataEmbedding_wo_temp
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import numpy as np


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

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Dense(configs.d_model, configs.c_out, has_bias=True)
        )

    def construct(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                  enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # MindSpore Tensor Initialization
        x_mark_enc = Tensor(np.zeros((x_enc.shape[0], x_enc.shape[1], 4)), ms.float32)
        x_dec = Tensor(np.zeros((x_enc.shape[0], 48 + 720, x_enc.shape[2])), ms.float32)
        x_mark_dec = Tensor(np.zeros((x_enc.shape[0], 48 + 720, 4)), ms.float32)

        # Decomposition
        mean = ops.ReduceMean(keep_dims=True)(x_enc, 1).repeat(1, self.pred_len, 1)
        zeros = ops.Zeros()((x_dec.shape[0], self.pred_len, x_dec.shape[2]), ms.float32)
        seasonal_init, trend_init = self.decomp(x_enc)

        # Decoder Input
        trend_init = ops.Concat(1)((trend_init[:, -self.label_len:, :], mean))
        seasonal_init = ops.Concat(1)((seasonal_init[:, -self.label_len:, :], zeros))

        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # Decoder
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)

        # Final Output
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
