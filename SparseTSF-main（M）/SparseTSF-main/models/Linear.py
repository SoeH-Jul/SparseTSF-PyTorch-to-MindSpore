import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context, Parameter
import numpy as np


class Model(nn.Cell):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Define the linear layer
        self.Linear = nn.Dense(self.seq_len, self.pred_len, has_bias=True)

        # Initialize weights to a constant value for visualization
        weight_init = Tensor((1 / self.seq_len) * np.ones([self.pred_len, self.seq_len]), ms.float32)
        self.Linear.weight = Parameter(weight_init, requires_grad=True)

    def construct(self, x):
        # x: [Batch, Input length, Channel]
        x = ops.Transpose()(x, (0, 2, 1))  # Permute to match input format
        x = self.Linear(x)
        x = ops.Transpose()(x, (0, 2, 1))  # Permute back to original format
        return x  # [Batch, Output length, Channel]


if __name__ == "__main__":
    # Define configuration class
    class Configs:
        def __init__(self):
            self.seq_len = 96
            self.pred_len = 48


    # Set execution context
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # Initialize model and configurations
    configs = Configs()
    model = Model(configs)

    # Test input
    batch_size = 32
    channels = 7
    x = Tensor(np.random.randn(batch_size, configs.seq_len, channels), ms.float32)

    # Forward pass
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
