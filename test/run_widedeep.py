from torch.utils.data import DataLoader
from data_loader import load_data
from models.WideDeep.network import WideDeep
from config.TrainConfig import *

from trainer.MultiTrainer import MultiTrainer
from trainer.MultiTrainerWeightingLoss import MultiTrainerWeightingLoss
import loss_weighting_strategy.EW as EW_strategy
import loss_weighting_strategy.UW as UW_strategy
import loss_weighting_strategy.DWA as DWA_strategy


def multi_test():
    net = WideDeep(
        num_of_dense_feature=num_of_dense_feature,
        sparse_features_val_num=sparse_features_val_num,
        deep_dropout=0,
        embed_dim=8,  # 用于控制稀疏特征经过Embedding层后的稠密特征大小
        hidden_layers=[256, 128, 64],
        output_dim=num_of_labels
    )

    trainer = MultiTrainer(model=net, config=widedeep_config)
    trainer.train(train_dataset)
    trainer.test(test_dataset)


def multi_weighting_test():
    model_args_dict = {
        'num_of_dense_feature': num_of_dense_feature,
        'sparse_features_val_num': sparse_features_val_num,
        'deep_dropout': 0,
        'embed_dim': 8,  # 用于控制稀疏特征经过Embedding层后的稠密特征大小
        'hidden_layers': [256, 128, 64],
        'output_dim': num_of_labels
    }

    weight_args_dict = DWA_strategy.default_args_dict

    optim_args_dict = {
        'optim': 'adam',
        'lr': widedeep_config['lr'],
        # 'weight_decay': widedeep_config['l2_regularization']
    }

    weighting_trainer = MultiTrainerWeightingLoss(
        model=WideDeep,
        weighting=DWA_strategy.DWA,
        config=widedeep_config,
        model_args_dict=model_args_dict,
        weight_args_dict=weight_args_dict,
        optim_args_dict=optim_args_dict
    )

    weighting_trainer.train(train_dataset)
    weighting_trainer.test(test_dataset)


if __name__ == '__main__':
    widedeep_config = \
        {
            'model_name': 'widedeep',
            'num_epoch': 25,
            'batch_size': 256,
            'lr': 1e-4,
            'l2_regularization': 1e-4,
        }

    train_dataset, test_dataset = load_data.load_dataset()

    multi_weighting_test()
