import torch
from torch.utils.data import DataLoader

from config.TrainConfig import *
from sklearn.metrics import classification_report, roc_auc_score


class MultiTrainer:

    def __init__(self, model, predict_by_model, config):
        self.model = model
        self.config = config
        self.predict_by_model = predict_by_model

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])

        for epoch in range(self.config['num_epoch']):
            epoch_loss = 0
            batch_num = 0
            self.model.train()
            for batch, train_data in enumerate(train_loader):

                x_data, y_data = train_data
                outputs = self.predict_by_model(x_data)

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

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

        with torch.no_grad():
            # 将batch_size设为测试集大小，以下循环只做一次
            for data in test_loader:
                x, y = data
                outputs = self.predict_by_model(x)

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
