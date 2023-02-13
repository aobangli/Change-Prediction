from config.TrainConfig import *
from data_loader import load_data
from models.LinearModel import LinearModel
from trainer.MultiTrainer import MultiTrainer
from trainer.MultiTrainerWeightingLoss import MultiTrainerWeightingLoss
import loss_weighting_strategy.EW as EW_strategy
import loss_weighting_strategy.UW as UW_strategy
import loss_weighting_strategy.DWA as DWA_strategy


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
def multi_test():

    num_of_features = len(load_data.feature_list)
    net = LinearModel(num_of_features, num_of_labels)

    trainer = MultiTrainer(model=net, config=dnn_config)
    trainer.train(train_dataset)
    trainer.test(test_dataset)


def multi_weighting_test():
    model_args_dict = {
        'input_dim': len(load_data.feature_list),
        'output_dim': num_of_labels
    }

    weight_args_dict = UW_strategy.default_args_dict

    optim_args_dict = {
        'optim': 'adam',
        'lr': dnn_config['lr'],
        # 'weight_decay': dnn_config['l2_regularization']
    }

    weighting_trainer = MultiTrainerWeightingLoss(
        model=LinearModel,
        weighting=UW_strategy.UW,
        config=dnn_config,
        model_args_dict=model_args_dict,
        weight_args_dict=weight_args_dict,
        optim_args_dict=optim_args_dict
    )

    weighting_trainer.train(train_dataset)
    weighting_trainer.test(test_dataset)


if __name__ == '__main__':
    dnn_config = \
        {
            'model_name': 'dnn',
            'output_dim': num_of_labels,
            'num_epoch': 30,
            'batch_size': 256,
            'lr': 1e-4,
            # 'l2_regularization': 1e-5,
        }

    train_dataset, test_dataset = load_data.load_dataset()

    multi_weighting_test()
