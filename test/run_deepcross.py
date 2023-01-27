import torch
from torch.utils.data import DataLoader

from models.DeepCross.trainer import Trainer
from models.DeepCross.network import DeepCross
from config.TrainConfig import *

from data_loader import load_data
from MultiTrainer import MultiTrainer


def test():
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

    correct = 0
    merged = 0
    abandoned = 0
    total = 0
    with torch.no_grad():  # 表示下面的计算不需要计算图和反向求导
        for data in test_loader:
            x, y = data
            outputs = net(x)
            # _, predicted = torch.max(outputs.data, 1)
            # 如果输出的预测值大于0.5，则predicted为1，否则为0
            predicted = torch.where(outputs.data > 0.5, 1, 0)
            predicted = predicted.view(-1)
            merged += (predicted == 1).sum().item()
            abandoned += (predicted == 0).sum().item()
            total += y.size(0)
            correct += (predicted == y).sum().item()  # 如果预测值和真实值相同， 则为true=1,  求和

    print("预测为阳性的样本数：%d" % merged)
    print("预测为阴性的样本数：%d" % abandoned)
    print("预测正确的样本数：%d" % correct)
    print("总样本数：%d" % total)
    print('Accuracy : %d %%' % (100 * correct / total))


def predict_by_deepcross(x):
    return net(x)


if __name__ == "__main__":
    deepcross_config = \
        {
            'deep_layers': [256, 128, 64, 32],  # 设置Deep模块的隐层大小
            'num_cross_layers': 4,  # cross模块的层数
            'output_dim': num_of_labels,
            'num_epoch': 10,
            'batch_size': 256,
            'lr': 1e-4,
            'l2_regularization': 1e-4,
            'device_id': 0,
            'use_cuda': False,
        }

    train_dataset, test_dataset = load_data.load_dataset(target_labels)

    net = DeepCross(deepcross_config, dense_features_cols=dense_features_cols,
                    sparse_features_cols=sparse_features_cols,
                    sparse_features_col_num=sparse_features_col_num)

    trainer = MultiTrainer(model=net, predict_by_model=predict_by_deepcross, config=deepcross_config)
    trainer.train(train_dataset)
    trainer.test(test_dataset)

    # trainer = Trainer(model=net, config=deepcross_config)
    # # 训练
    # trainer.train(train_dataset)
    # test()
