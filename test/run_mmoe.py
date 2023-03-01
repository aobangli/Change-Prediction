# -*- coding: utf-8 -*-
import numpy as np

import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from models.mmoe import MMOE

from config.TrainConfig import *
from data_loader import load_data
from trainer.MultiTrainerWeightingLoss import MultiTrainerWeightingLoss
import loss_weighting_strategy.EW as EW_strategy
import loss_weighting_strategy.UW as UW_strategy
import loss_weighting_strategy.DWA as DWA_strategy


def transfer_task_types():
    types = []
    for label_type in label_types:
        if label_type == TaskType.Binary_Classification:
            types.append('binary')
        elif label_type == TaskType.Regression:
            types.append('regression')
        elif label_type == TaskType.Multiple_Classification:
            types.append('multiclass')
        else:
            raise ValueError("task must be binary, multiclass or regression, {} is illegal".format(label_type))
    return types


def multi_weighting_test():
    df = load_data.prepare_dataframe()

    all_features = [SparseFeat(feat, vocabulary_size=(df[feat].max() + 1).astype(np.int_), embedding_dim=4)
                    for feat in sparse_features_cols] + [DenseFeat(feat, 1, ) for feat in dense_features_cols]

    task_types = transfer_task_types()

    model_args_dict = {
        'dnn_feature_columns': all_features,
        'task_types': task_types,
        'task_names': target_labels,
        'task_out_dims': task_out_dims
    }

    weight_args_dict = EW_strategy.default_args_dict

    optim_args_dict = {
        'optim': 'adam',
        'lr': mmoe_config['lr'],
        # 'weight_decay': widedeep_config['l2_regularization']
    }

    weighting_trainer = MultiTrainerWeightingLoss(
        model=MMOE,
        weighting=EW_strategy.EW,
        config=mmoe_config,
        model_args_dict=model_args_dict,
        weight_args_dict=weight_args_dict,
        optim_args_dict=optim_args_dict
    )

    # 根据制定的特征顺序对数据集列重新排序，以适配模型输入
    feature_name = get_feature_names(all_features)

    train_dataset = load_data.MyDataset(train_df, reordered_feature_list=feature_name)
    test_dataset = load_data.MyDataset(test_df, reordered_feature_list=feature_name)

    weighting_trainer.train(train_dataset)
    weighting_trainer.test(test_dataset)


if __name__ == "__main__":
    mmoe_config = {
        'model_name': 'mmoe',
        'num_epoch': 25,
        'batch_size': 256,
        'lr': 1e-4,
        'l2_regularization': 1e-4,
    }

    train_df, test_df = load_data.load_splited_dataframe()

    multi_weighting_test()
