from torch.utils.data import DataLoader

from models.DeepCross.network import DeepCross
from config.TrainConfig import *

from data_loader import load_data
from trainer.MultiTrainer import MultiTrainer
from trainer.MultiTrainerWeightingLoss import MultiTrainerWeightingLoss
import loss_weighting_strategy.EW as EW_strategy
import loss_weighting_strategy.UW as UW_strategy
import loss_weighting_strategy.DWA as DWA_strategy


# 多任务训练并测试
def multi_test():
    net = DeepCross(
        num_of_dense_feature=num_of_dense_feature,
        sparse_features_val_num=sparse_features_val_num,
        deep_layers=[256, 128, 64, 32],  # 设置Deep模块的隐层大小
        num_cross_layers=4,  # cross模块的层数
        output_dim=num_of_labels
    )

    trainer = MultiTrainer(model=net, config=deepcross_config)
    trainer.train(train_dataset)
    trainer.test(test_dataset)


# 多任务、loss加权训练并测试
def multi_weighting_test():
    model_args_dict = {
        'num_of_dense_feature': num_of_dense_feature,
        'sparse_features_val_num': sparse_features_val_num,
        'deep_layers': [256, 128, 64, 32],  # 设置Deep模块的隐层大小
        'num_cross_layers': 4,  # cross模块的层数
        'output_dim': num_of_labels
    }

    weight_args_dict = DWA_strategy.default_args_dict

    optim_args_dict = {
        'optim': 'adam',
        'lr': deepcross_config['lr'],
        # 'weight_decay': deepcross_config['l2_regularization']
    }

    weighting_trainer = MultiTrainerWeightingLoss(
        model=DeepCross,
        weighting=DWA_strategy.DWA,
        config=deepcross_config,
        model_args_dict=model_args_dict,
        weight_args_dict=weight_args_dict,
        optim_args_dict=optim_args_dict
    )

    weighting_trainer.train(train_dataset)
    weighting_trainer.test(test_dataset)


if __name__ == "__main__":
    deepcross_config = \
        {
            'model_name': 'deepcross',
            'num_epoch': 30,
            'batch_size': 256,
            'lr': 1e-4,
            'l2_regularization': 1e-5,
        }

    train_dataset, test_dataset = load_data.load_dataset()
    multi_weighting_test()
