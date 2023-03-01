from tab_transformer_pytorch import TabTransformer

from torch.utils.data import DataLoader
from data_loader import load_data
from config.TrainConfig import *

from trainer.MultiTrainer import MultiTrainer
from trainer.MultiTrainerWeightingLoss import MultiTrainerWeightingLoss
import loss_weighting_strategy.EW as EW_strategy
import loss_weighting_strategy.UW as UW_strategy
import loss_weighting_strategy.DWA as DWA_strategy


def multi_test():
    # cont_mean_std = torch.randn(10, 2)

    net = TabTransformer(
        # categories=(10, 5, 6, 5, 8),  # tuple containing the number of unique values within each category
        categories=tuple(sparse_features_val_num),
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

    # x_categ = torch.randint(0, 5, (
    # 1, 5))  # category values, from 0 - max number of categories, in the order as passed into the constructor above
    # x_cont = torch.randn(1, 10)  # assume continuous values are already normalized individually
    #
    # pred = model(x_categ, x_cont)  # (1, 1)
    # print(pred)

    trainer = MultiTrainer(model=net, config=tabtransformer_config)

    trainer.train(train_dataset)
    trainer.test(test_dataset)


def multi_weighting_test():
    model_args_dict = {
        'categories': tuple(sparse_features_val_num),
        'num_continuous': num_of_dense_feature,  # number of continuous values
        'dim': 32,  # dimension, paper set at 32
        'dim_out': num_of_labels,  # binary prediction, but could be anything
        'depth': 6,  # depth, paper recommended 6
        'heads': 8,  # heads, paper recommends 8
        'attn_dropout': 0.1,  # post-attention dropout
        'ff_dropout': 0.1,  # feed forward dropout
        'mlp_hidden_mults': (4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        'mlp_act': nn.ReLU(),  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        # continuous_mean_std=cont_mean_std  # (optional) - normalize the continuous values before layer norm
    }

    weight_args_dict = EW_strategy.default_args_dict

    optim_args_dict = {
        'optim': 'adam',
        'lr': tabtransformer_config['lr'],
        # 'weight_decay': widedeep_config['l2_regularization']
    }

    weighting_trainer = MultiTrainerWeightingLoss(
        model=TabTransformer,
        weighting=EW_strategy.EW,
        config=tabtransformer_config,
        model_args_dict=model_args_dict,
        weight_args_dict=weight_args_dict,
        optim_args_dict=optim_args_dict
    )

    weighting_trainer.train(train_dataset)
    weighting_trainer.test(test_dataset)


if __name__ == "__main__":
    tabtransformer_config = \
        {
            'model_name': 'tabtransformer',
            'num_epoch': 30,
            'batch_size': 256,
            'lr': 1e-4,
        }

    train_dataset, test_dataset = load_data.load_dataset()

    multi_weighting_test()
