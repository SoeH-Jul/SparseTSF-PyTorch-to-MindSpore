from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset
from mindspore import Tensor

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    # 选择数据集和相关参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # 初始化数据集
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))

    # 数据集转换为 GeneratorDataset
    def generator():
        for i in range(len(data_set)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = data_set[i]
            yield seq_x.asnumpy(), seq_y.asnumpy(), seq_x_mark.asnumpy(), seq_y_mark.asnumpy()

    # 创建 GeneratorDataset
    dataset = GeneratorDataset(generator, ["seq_x", "seq_y", "seq_x_mark", "seq_y_mark"])

    # 设置批次大小、是否洗牌等参数
    dataset = dataset.batch(batch_size, drop_remainder=drop_last)

    # 如果需要设置乱序，传入 shuffle
    if shuffle_flag:
        dataset = dataset.shuffle(buffer_size=10000)

    return data_set, dataset
