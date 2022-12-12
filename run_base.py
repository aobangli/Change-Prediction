import torch
from torch import nn
from torch.utils.data import DataLoader
import load_data
from models.LinearModel import LinearModel


# def run():
#     feature_list = load_data.feature_list
#     n = len(feature_list)
#
#     net = LinearModel(n)
#
#     learning_rate = 0.001
#     num_of_epoch = 1
#     batch_size = 16
#
#     train_dataset, test_dataset = load_data.load_dataset()
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train():
    loss_F = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(num_of_epoch):
        sum_loss = 0
        batch_num = 0
        net.train()
        for index, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            x_data, y_data = train_data
            outputs = net(x_data)
            loss = loss_F(outputs, y_data)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if index % 10 == 0:
                print('epoch = ', epoch + 1, 'loss = ', loss.item(), 'index = ', index + 1)
            batch_num += 1
        print(sum_loss / batch_num)


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 表示下面的计算不需要计算图和反向求导
        for data in test_loader:
            x, y = data
            outputs = net(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()  # 如果预测值和真实值相同， 则为true=1,  求和

    print(correct)
    print(total)
    print('Accuracy : %d %%' % (100 * correct / total))


if __name__ == '__main__':
    feature_list = load_data.feature_list
    n = len(feature_list)

    net = LinearModel(n)

    learning_rate = 0.001
    num_of_epoch = 10
    batch_size = 128

    train_dataset, test_dataset = load_data.load_dataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train()
    test()
