import os
import time

#import epoch
import mindspore
import numpy as np
import warnings
from mindspore import context, Tensor, ops
import mindspore.nn as nn
from mindspore.experimental.optim import lr_scheduler
from mindspore.train import Model
from mindspore.nn import Adam, Optimizer
from mindspore.train.callback import EarlyStopping
from mindspore import save_checkpoint

from mindspore import nn


from mindspore.experimental import optim

from utils.tools import visual, test_params_flop,EarlyStopping,adjust_learning_rate

# 假设你已经定义了模型和优化器 model_optim
learning_rate = 0.1
decay_rate = 0.96
decay_steps = 1000

# 使用 ExponentialDecayLR 作为调度器
scheduler = nn.ExponentialDecayLR(learning_rate, decay_rate, decay_steps)

# 或者使用 PiecewiseDecay
boundaries = [10000, 20000]  # 例如，学习率衰减的边界
values = [0.1, 0.05, 0.01]  # 学习率的不同值
# 创建 PiecewiseDecay 学习率调度器



from models.FEDformer import model

# 假设 model_optim 已经是一个模型优化器
learning_rate = 0.001
decay_rate = 0.9
decay_steps = 100
step_per_epoch = 100  # 假设每个 epoch 有 100 个训练步骤
decay_epoch = 10      # 每 10 个 epoch 衰减一次学习率

# 使用 ExponentialDecay 学习率调度器
lr_schedule = nn.exponential_decay_lr(learning_rate, decay_rate, decay_steps,step_per_epoch, decay_epoch)

# 选择优化器，假设模型和参数已经定义
optimizer = Adam(model.trainable_params(), learning_rate=lr_schedule)

# 训练过程中使用优化器



import matplotlib.pyplot as plt

from data_provider.data_factory import data_provider
from models import Autoformer, Transformer, Informer, DLinear, Linear, PatchTST, SparseTSF
from utils.metrics import metric
from utils.tools import adjust_learning_rate, visual, test_params_flop

# 忽略警告
warnings.filterwarnings('ignore')


class Exp_Main(nn.Cell):
    def __init__(self, args):
        super(Exp_Main, self).__init__()
        self.args = args
        # 初始化模型
        self.model = self._build_model()
        # 设置设备
        self.device = context.get_context("device_target")
        context.set_context(device_target="CPU")

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF
        }
        model = model_dict[self.args.model].Model(self.args)

        # 若使用多GPU，则进行DataParallel
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 加载数据集
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 使用Adam优化器

        model_optim = Adam(self.model.trainable_params(), learning_rate=self.args.learning_rate)

        return model_optim


    def _select_criterion(self):
        # 根据损失函数类型选择
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.set_train(False)  # 设置为验证模式
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = Tensor(batch_x, dtype=mindspore.float32)
            batch_y = Tensor(batch_y, dtype=mindspore.float32)

            batch_x_mark = Tensor(batch_x_mark, dtype=mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, dtype=mindspore.float32)

            # decoder 输入
            # dec_inp = Tensor(np.zeros(batch_yl:, -self.args.pred_len:, :l.shape, dtype=np.float32), dtype=mindspore.float32)
            # dec_inp = ops.Concat(1)((batch_y[:,:self.args.label_len, :l,dec_inp))

            # 编码器-解码器
            if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark,batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark,batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            pred = outputs.asnumpy()
            true = batch_y.asnumpy()

            loss = criterion(Tensor(pred), Tensor(true))
            total_loss.append(loss.asnumpy())

        self.model.set_train(True)
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
           context.set_context(enable_auto_mixed_precision = True)
        # 创建学习率调度器
        total_steps = self.args.train_epochs * train_steps
        min_lr = self.args.learning_rate / 10  # 最小学习率
        loss_fn = nn.WithLossCell(self.model, criterion)
        train_net = nn.TrainOneStepCell(loss_fn, model_optim)
        #设置训练模式
        train_net.set_train()




        for epoch in range(self.args.train_epochs):
            #adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            iter_count = 0
            train_loss = []
            self.model.set_train(True)  # 设置为训练模式
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                #print(f"Processing batch {i + 1}")

                batch_x = Tensor(batch_x, dtype=mindspore.float32)
                batch_y = Tensor(batch_y, dtype=mindspore.float32)

                batch_x_mark = Tensor(batch_x_mark, dtype=mindspore.float32)
                batch_y_mark = Tensor(batch_y_mark, dtype=mindspore.float32)
                # decoder 输入
                # dec_inp = Tensor(
                #     np.zeros_like(batch_y[:, -self.args.pred_len:, :])).float()  # 确保 dtype 为 float32
                # dec_inp = ops.Concat(axis=1)([batch_y[:, :self.args.label_len, :], dec_inp])

                # 编码器-解码器
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark,batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark,batch_y_mark)

                # 打印模型的输出形状
                #print(f"Model output shape: {outputs.shape}")
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # print(f"Outputs shape: {outputs.shape}")
                # print(f"True shape: {batch_y.shape}")

                # 计算损失
                loss = train_net(batch_x, batch_y)  # 注意这里只有 batch_x 和 batch_y
                train_loss.append(loss.asnumpy())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.asnumpy()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()  # 确保loss在此之前已经计算
                # 对梯度进行裁剪
                # # 添加梯度裁剪的值
                # forward_fn = self.model
                # grad_fn = mindspore.grad(forward_fn, grad_position=None, weights=model.trainable_params(), has_aux=True)
                # grads = grad_fn(mindspore.ops.unsqueeze(batch_x, dim=0))
                # grad_clip = mindspore.ops.clip_by_value(grads, clip_value_min=-0.1, clip_value_max=0.1)
                #
                # train_net.set_grad(grad_clip)
                #model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                  print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        self.test(setting)
        best_model_path = f"{path}/checkpoint.ckpt"
        param_dict=self.model.load_state_dict(mindspore.load_checkpoint(best_model_path))
        mindspore.load_param_into_net(self.model, param_dict)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            # Load checkpoint
            self.model.set_train(False)
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.ckpt')
            param_dict = mindspore.load_checkpoint(checkpoint_path)
            mindspore.load_param_into_net(self.model, param_dict)

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.set_train(False)#mindspore中不需要通过set_eval设置评估模式
        #self.model.set_eval()  # Set model to eval mode
        #with mindspore.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = Tensor(batch_x, dtype=mindspore.float32)
            batch_y = Tensor(batch_y, dtype=mindspore.float32)

            batch_x_mark = Tensor(batch_x_mark, dtype=mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, dtype=mindspore.float32)

            # # decoder input
            # dec_inp = Tensor(np.zeros_like(batch_y[:, -self.args.pred_len:, :])).float()
            # dec_inp = mindspore.ops.Concat(1)((batch_y[:, :self.args.label_len, :], dec_inp))

            # if self.args.use_amp:
            #     with mindspore.context.autotune(True):
            #         if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
            #             outputs = self.model(batch_x)
            #         else:
            #             if self.args.output_attention:
            #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            #             else:
            #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # else:
            if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                outputs = self.model(batch_x)
                # else:
                #     if self.args.output_attention:
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                #     else:
                #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            self.device = context.get_context("device_target")
            context.set_context(device_target="CPU")
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            #batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

            # 先移动到目标设备
            #batch_y = batch_y.to(self.device)

            # 然后转换数据类型
            batch_y = batch_y.astype(mindspore.float32)

            pred = outputs.asnumpy()
            true = batch_y.asnumpy()

            preds.append(pred)
            trues.append(true)
            inputx.append(batch_x.asnumpy())
            if i % 20 == 0:
                input = batch_x.asnumpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # If the test requires the parameter FLOP analysis
        # if self.args.test_flop:
        #     test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
        #     exit()

        # Concatenate predictions, true values, and inputs for later evaluation
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        with open("result.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
            f.write('\n\n')

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            checkpoint_path = path + '/' + 'checkpoint.ckpt'
            param_dict = mindspore.load_checkpoint(checkpoint_path)
            mindspore.load_param_into_net(self.model, param_dict)

        preds = []
        self.model.set_train(False)
        #self.model.set_eval()  # Set model to eval mode
        #with mindspore.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            batch_x = Tensor(batch_x, dtype=mindspore.float32)
            batch_y = Tensor(batch_y, dtype=mindspore.float32)

            batch_x_mark = Tensor(batch_x_mark, dtype=mindspore.float32)
            batch_y_mark = Tensor(batch_y_mark, dtype=mindspore.float32)

            # decoder input
            # dec_inp = Tensor(np.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]),
            #                      dtype=mindspore.float32).to(batch_y.device)
            # dec_inp = mindspore.ops.Concat(1)((batch_y[:, :self.args.label_len, :], dec_inp))

            #if self.args.use_amp:
                #with mindspore.context.autotune(True):
                    #if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                        #outputs = self.model(batch_x)
                    #else:
                        #if self.args.output_attention:
                            ##outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        #else:
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                outputs = self.model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark,batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark,batch_y_mark)

                pred = outputs.asnumpy()  # .squeeze()
                preds.append(pred)

            preds = np.array(preds)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

            # Save results
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path + 'real_prediction.npy', preds)

            return
