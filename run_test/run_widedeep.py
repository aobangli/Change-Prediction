import torch
from torch.utils.data import DataLoader
import load_data
from models.WideDeep.network import WideDeep
from models.WideDeep.trainer import Trainer
from load_data import dense_features_cols

widedeep_config = \
    {
        'deep_dropout': 0,
        'embed_dim': 8,  # 用于控制稀疏特征经过Embedding层后的稠密特征大小
        'hidden_layers': [256, 128, 64],
        'num_epoch': 15,
        'batch_size': 256,
        'lr': 1e-3,
        'l2_regularization': 1e-4,
        'device_id': 0,
        'use_cuda': False,
    }


def test():
    correct = 0
    merged = 0
    abandoned = 0
    total = 0
    with torch.no_grad():  # 表示下面的计算不需要计算图和反向求导
        for data in test_loader:
            x, y = data
            outputs = wideDeep(x)
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


if __name__ == '__main__':
    feature_list = load_data.feature_list
    # n = len(feature_list)

    train_dataset, test_dataset = load_data.load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=widedeep_config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=widedeep_config['batch_size'], shuffle=False)

    wideDeep = WideDeep(config=widedeep_config, dense_features_cols=dense_features_cols, sparse_features_cols=[2, 2, 2])

    # # 实例化模型训练器
    trainer = Trainer(model=wideDeep, config=widedeep_config)
    # 训练
    trainer.train(train_dataset)

    test()



