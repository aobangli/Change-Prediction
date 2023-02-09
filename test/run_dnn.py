import torch
from torch import nn
from torch.utils.data import DataLoader

from config.TrainConfig import *
from data_loader import load_data
from models.LinearModel import LinearModel
from MultiTrainer import MultiTrainer


# def train():
#     # loss_F = nn.CrossEntropyLoss()
#     loss_F = nn.BCELoss()
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#
#     for epoch in range(num_of_epoch):
#         sum_loss = 0
#         batch_num = 0
#         net.train()
#         for index, train_data in enumerate(train_loader):
#             optimizer.zero_grad()
#             x_data, y_data = train_data
#             outputs = net(x_data)
#             loss = loss_F(outputs.view(-1), y_data)
#             loss.backward()
#             optimizer.step()
#
#             sum_loss += loss.item()
#             if index % 10 == 0:
#                 print('epoch = ', epoch + 1, 'loss = ', loss.item(), 'index = ', index + 1)
#             batch_num += 1
#         print(sum_loss / batch_num)
#
#
# def test():
#     correct = 0
#     merged = 0
#     abandoned = 0
#     total = 0
#     with torch.no_grad():  # 表示下面的计算不需要计算图和反向求导
#         for data in test_loader:
#             x, y = data
#             outputs = net(x)
#             # _, predicted = torch.max(outputs.data, 1)
#             # 如果输出的预测值大于0.5，则predicted为1，否则为0
#             predicted = torch.where(outputs.data > 0.5, 1, 0)
#             predicted = predicted.view(-1)
#             merged += (predicted == 1).sum().item()
#             abandoned += (predicted == 0).sum().item()
#             total += y.size(0)
#             correct += (predicted == y).sum().item()  # 如果预测值和真实值相同， 则为true=1,  求和
#
#     print("预测为阳性的样本数：%d" % merged)
#     print("预测为阴性的样本数：%d" % abandoned)
#     print("预测正确的样本数：%d" % correct)
#     print("总样本数：%d" % total)
#     print('Accuracy : %d %%' % (100 * correct / total))


def predict_by_dnn(x):
    return net(x)


if __name__ == '__main__':

    dnn_config = \
        {
            'model_name': 'dnn',
            'output_dim': num_of_labels,
            'num_epoch': 30,
            'batch_size': 256,
            'lr': 1e-4,
        }

    feature_list = load_data.feature_list
    num_of_features = len(feature_list)

    net = LinearModel(num_of_features, num_of_labels)

    train_dataset, test_dataset = load_data.load_dataset()

    trainer = MultiTrainer(model=net, predict_by_model=predict_by_dnn, config=dnn_config)
    trainer.train(train_dataset)
    trainer.test(test_dataset)
