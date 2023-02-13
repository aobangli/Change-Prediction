import numpy as np
from torch.utils.data import DataLoader

from config.TrainConfig import *
from sklearn.metrics import classification_report, roc_auc_score, r2_score

from torch.utils.tensorboard import SummaryWriter
import time

from config.model_input_func import func_dict


# def print_loss(epoch, loss):
#     # 可以直接使用python的with语法，自动调用close方法
#     with SummaryWriter(log_dir='/Users/aobang/PycharmProjects/tensorboard_logs', comment='train') as writer:
#         # writer.add_histogram('his/x', x, epoch)
#         # writer.add_histogram('his/y', y, epoch)
#         # writer.add_scalar('data/x', x, epoch)
#         # writer.add_scalar('data/y', y, epoch)
#         writer.add_scalar('data/loss', loss, epoch)
#         # writer.add_scalars('data/data_group', {'x': x, 'y': y}, epoch)


class MultiTrainerWeightingLoss:

    def __init__(self, model, weighting, config, model_args_dict, weight_args_dict, optim_args_dict, device='cpu'):
        # self.model = model
        self.weighting = weighting
        self.config = config
        self.model_args_dict = model_args_dict
        self.weight_args_dict = weight_args_dict
        self.optim_args_dict = optim_args_dict

        self.predict_by_model = func_dict[self.config["model_name"]]

        self.task_num = num_of_labels
        self.device = device

        self._prepare_model(weighting, model)
        self._prepare_optimizer(optim_args_dict)

    def _prepare_optimizer(self, optim_param):
        optim_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            # 'adagrad': torch.optim.Adagrad,
            # 'rmsprop': torch.optim.RMSprop,
        }

        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.mtl_model.parameters(), **optim_arg)

    def _prepare_model(self, weighting, architecture):

        class MTLmodel(architecture, weighting):
            def __init__(self, model_args_dict, device):
                super(MTLmodel, self).__init__(**model_args_dict)
                self.task_num = num_of_labels
                self.device = device
                self.init_param()

        self.mtl_model = MTLmodel(self.model_args_dict, self.device)

    def train(self, train_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)

        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr']
        #                              # ,weight_decay=self.config['l2_regularization']
        #                              )

        time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        # 使用tensorboard画loss图
        writer = SummaryWriter(f'/Users/aobang/PycharmProjects/tensorboard_logs/{self.config["model_name"]}_{time_str}')
        # loss的横坐标
        loss_index = 0
        # weight的横坐标
        weight_index = 0

        # # 记录优化过的不同任务loss的权值
        # batch_weight = np.zeros([self.task_num, self.config['num_epoch'], self.config['batch_size']])
        # 缓存每个epoch中每个任务的loss，某些动态权重策略需要参考之前的loss来修改当前的权重
        self.mtl_model.train_loss_buffer = np.zeros([self.task_num, self.config['num_epoch']])

        for epoch in range(self.config['num_epoch']):
            # 记录当前epoch每个任务的平均loss
            epoch_losses = np.zeros([self.task_num])
            # 每个epoch加权后的loss
            epoch_weighted_loss = 0
            # batch数量计数器
            batch_num = 0
            # 样本数量计数器
            sample_num = 0

            self.mtl_model.epoch = epoch
            self.mtl_model.train()
            for batch, train_data in enumerate(train_loader):

                x_data, y_data = train_data
                outputs = self.predict_by_model(x_data, self.mtl_model)

                assert outputs.shape[1] == num_of_labels == y_data.shape[1], "输出维数应该与标签个数相等！"

                all_labels_loss = torch.zeros(num_of_labels)
                for label_index, (label, loss_func) in enumerate(zip(target_labels, loss_functions_by_label)):
                    output = outputs[:, label_index]
                    loss = loss_func(output.view(-1), y_data[:, label_index])
                    # 记录当前batch的loss，在权重策略中调整并执行backward
                    all_labels_loss[label_index] = loss
                    # 将每个样本的loss加起来，后面计算整个epoch的平均值存入train_loss_buffer
                    batch_size = output.size()[0]
                    epoch_losses[label_index] += loss.item() * batch_size
                    sample_num += batch_size
                    # 将每个task的loss加入tensorboard作图，每个batch为一个数据点
                    writer.add_scalar(f'loss/{label}', loss.item(), loss_index)

                self.optimizer.zero_grad()
                weighted_loss, weight = self.mtl_model.backward(all_labels_loss, **self.weight_args_dict)
                self.optimizer.step()

                # batch_weight[:, epoch, batch] = weight
                for label_index, label in enumerate(target_labels):
                    label_weight = weight[label_index]
                    if batch % 10 == 0:
                        print(f'label: {label}  weight: {label_weight}')
                    writer.add_scalar(f'weight/{label}', label_weight, weight_index)
                weight_index += 1

                writer.add_scalar('loss/weighted_loss', weighted_loss, loss_index)
                loss_index += 1

                if batch % 10 == 0:
                    print('epoch = ', epoch + 1, 'loss = ', weighted_loss, 'batch = ', batch + 1)

                epoch_weighted_loss += weighted_loss
                batch_num += 1
            # 每个epoch的所有batch完成后执行的操作

            # 记录每个epoch的每个任务的平均loss
            for label_index in range(self.task_num):
                self.mtl_model.train_loss_buffer[label_index, epoch] = epoch_losses[label_index] / sample_num

            print('epoch = ', epoch + 1, 'epoch loss = ', epoch_weighted_loss / batch_num)
        writer.close()

    def test(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=False)

        with torch.no_grad():
            # 将batch_size设为测试集大小，以下循环只做一次
            for data in test_loader:
                x, y = data
                outputs = self.predict_by_model(x, self.mtl_model)

                assert outputs.shape[1] == num_of_labels == y.shape[1], "输出维数应该与标签个数相等！"

                for label_index, (label, label_type) in enumerate(zip(target_labels, label_types)):
                    print("#" * 10, "Label:", label, "#" * 10)
                    output = outputs[:, label_index]
                    if label_type == LabelType.Binary_Classification:
                        # 如果输出的预测值大于0.5，则predicted为1，否则为0
                        predicted = torch.where(output.data > 0.5, 1, 0)
                        predicted = predicted.view(-1)

                        print(classification_report(y[:, label_index], predicted))
                        print("auc = ", roc_auc_score(y[:, label_index], output.data))
                    elif label_type == LabelType.Multiple_Classification:
                        pass
                    elif label_type == LabelType.Regression:
                        y_true = y[:, label_index]
                        y_pred = output
                        print("mse = ", nn.MSELoss()(y_pred, y_true).item())
                        print("R2 score = ", r2_score(y_true, y_pred))
