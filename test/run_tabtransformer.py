import torch
from tab_transformer_pytorch import TabTransformer

from torch.utils.data import DataLoader
from data_loader import load_data
from config.TrainConfig import *
from sklearn.metrics import classification_report, roc_auc_score

from MultiTrainer import MultiTrainer


def train():
    train_loader = DataLoader(train_dataset, batch_size=tabtransformer_config['batch_size'], shuffle=True)
    # loss_F = nn.CrossEntropyLoss()
    # loss_F = nn.BCELoss()
    loss_F = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=tabtransformer_config['lr'])

    for epoch in range(tabtransformer_config['num_epoch']):
        sum_loss = 0
        batch_num = 0
        net.train()
        for index, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            x_data, y_data = train_data
            dense_input, sparse_inputs = x_data[:, :num_of_dense_feature], x_data[:, num_of_dense_feature:]
            outputs = net(sparse_inputs.long(), dense_input)
            loss = loss_F(outputs.view(-1), y_data.view(-1))
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            if index % 10 == 0:
                print('epoch = ', epoch + 1, 'loss = ', loss.item(), 'index = ', index + 1)
            batch_num += 1
        print(sum_loss / batch_num)


def test():
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

    correct = 0
    merged = 0
    abandoned = 0
    total = 0
    with torch.no_grad():  # 表示下面的计算不需要计算图和反向求导
        for data in test_loader:
            x, y = data
            dense_input, sparse_inputs = x[:, :num_of_dense_feature], x[:, num_of_dense_feature:]
            outputs = net(sparse_inputs.long(), dense_input)
            # _, predicted = torch.max(outputs.data, 1)

            # 如果输出的预测值大于0.5，则predicted为1，否则为0
            predicted = torch.where(outputs.data > 0.5, 1, 0)
            predicted = predicted.view(-1)
            merged += (predicted == 1).sum().item()
            abandoned += (predicted == 0).sum().item()
            total += y.size(0)
            correct += (predicted == y.view(-1)).sum().item()  # 如果预测值和真实值相同， 则为true=1,  求和

    print("预测为阳性的样本数：%d" % merged)
    print("预测为阴性的样本数：%d" % abandoned)
    print("预测正确的样本数：%d" % correct)
    print("总样本数：%d" % total)
    print('Accuracy : %d %%' % (100 * correct / total))


def multi_train():
    train_loader = DataLoader(train_dataset, batch_size=tabtransformer_config['batch_size'], shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=tabtransformer_config['batch_size'])

    for epoch in range(tabtransformer_config['num_epoch']):
        epoch_loss = 0
        batch_num = 0
        net.train()
        for batch, train_data in enumerate(train_loader):

            x_data, y_data = train_data
            outputs = predict_by_tabtransformer(x_data)

            assert outputs.shape[1] == num_of_labels == y_data.shape[1], "输出维数应该与标签个数相等！"

            all_labels_loss = 0
            for label_index in range(num_of_labels):
                output = outputs[:, label_index]
                loss_F = loss_functions_by_label[label_index]
                loss = loss_F(output.view(-1), y_data[:, label_index])
                all_labels_loss += loss

            optimizer.zero_grad()
            all_labels_loss.backward()
            optimizer.step()

            avg_loss = all_labels_loss.item() / num_of_labels

            epoch_loss += avg_loss
            if batch % 10 == 0:
                print('epoch = ', epoch + 1, 'loss = ', avg_loss, 'batch = ', batch + 1)
            batch_num += 1
        print('epoch = ', epoch + 1, 'avg loss = ', epoch_loss / batch_num)


def multi_test():
    test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

    with torch.no_grad():
        # 将batch_size设为测试集大小，以下循环只做一次
        for data in test_loader:
            x, y = data
            outputs = predict_by_tabtransformer(x)

            assert outputs.shape[1] == num_of_labels == y.shape[1], "输出维数应该与标签个数相等！"

            for label_index, (label, label_type) in enumerate(zip(target_labels, label_types)):
                print("#" * 10, "Label:", label, "#" * 10)
                output = outputs[:, label_index]
                if label_type == LabelType.Binary_Classification:
                    # 如果输出的预测值大于0.5，则predicted为1，否则为0
                    predicted = torch.where(output.data > 0.5, 1, 0)
                    predicted = predicted.view(-1)

                    print(classification_report(y[:, label_index], predicted))
                    print("auc = ", roc_auc_score(y[:, label_index], predicted))
                elif label_type == LabelType.Multiple_Classification:
                    pass
                elif label_type == LabelType.Regression:
                    criterion = nn.MSELoss()
                    print("mse loss = ", criterion(output, y[:, label_index]).item())


# 使用tabtransformer计算结果
def predict_by_tabtransformer(x):
    dense_input, sparse_inputs = x[:, :num_of_dense_feature], x[:, num_of_dense_feature:]
    outputs = net(sparse_inputs.long(), dense_input)
    return outputs


if __name__ == "__main__":
    tabtransformer_config = \
        {
            'model_name': 'tabtransformer',
            'num_epoch': 20,
            'batch_size': 256,
            'lr': 1e-4,
        }

    train_dataset, test_dataset = load_data.load_dataset()

    # cont_mean_std = torch.randn(10, 2)

    net = TabTransformer(
        # categories=(10, 5, 6, 5, 8),  # tuple containing the number of unique values within each category
        categories=tuple(sparse_features_col_num),
        num_continuous=num_of_dense_feature,  # number of continuous values
        dim=32,  # dimension, paper set at 32
        dim_out=num_of_labels,  # binary prediction, but could be anything
        depth=6,  # depth, paper recommended 6
        heads=8,  # heads, paper recommends 8
        attn_dropout=0.1,  # post-attention dropout
        ff_dropout=0.1,  # feed forward dropout
        mlp_hidden_mults=(4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        mlp_act=nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        # continuous_mean_std=cont_mean_std  # (optional) - normalize the continuous values before layer norm
    )

    trainer = MultiTrainer(model=net, predict_by_model=predict_by_tabtransformer, config=tabtransformer_config)

    trainer.train(train_dataset)
    trainer.test(test_dataset)

    # multi_train()
    # multi_test()

    # train()
    # test()

    # x_categ = torch.randint(0, 5, (
    # 1, 5))  # category values, from 0 - max number of categories, in the order as passed into the constructor above
    # x_cont = torch.randn(1, 10)  # assume continuous values are already normalized individually
    #
    # pred = model(x_categ, x_cont)  # (1, 1)
    # print(pred)
