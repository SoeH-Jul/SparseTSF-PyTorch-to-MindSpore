import os
import numpy as np
import mindspore
from mindspore import context
import mindspore.nn as nn

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if not self.args.use_multi_gpu:
                context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=self.args.gpu)
                print(f'Use GPU: {self.args.gpu}')
            else:
                context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL, device_num=len(self.args.devices.split(',')))
                context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=int(self.args.gpu))
                print(f'Use Multi-GPU: {self.args.devices}')
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
            print('Use CPU')
        return self.device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
