import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

from config.TrainConfig import *


def preprocess_features(df):
    # 进行编码  类别特征编码
    for feature in sparse_features_cols:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        df[feature] = df[feature].astype('float')

    # 数值特征归一化
    # mms = MinMaxScaler()
    # dense_feature_values = mms.fit_transform(df[dense_features_cols])

    # 数值特征标准化
    ss = StandardScaler()
    dense_feature_values = ss.fit_transform(df[dense_features_cols])

    df[dense_features_cols] = dense_feature_values
    return df[feature_list]


def preprocess_labels(df):
    for label, label_type in zip(target_labels, label_types):
        if label in regression_labels:
            if label_type == LabelType.Binary_Classification:
                df[[label]] = transform_label_to_binary_classification(df[[label]].copy())
            elif label_type == LabelType.Multiple_Classification:
                df[[label]] = transform_label_to_multi_classification(df[[label]].copy())
            elif label_type == LabelType.Regression:
                # 暂不做归一化
                # self.df[[label]] = MinMaxScaler().fit_transform(self.df[[label]])
                df[[label]] = StandardScaler().fit_transform(df[[label]])
    return df[target_labels]


def transform_label_to_binary_classification(df):
    for label in list(df.columns):
        threshold = binary_classification_label_threshold[label]
        df[label] = df[label].apply(lambda x: 1 if x > threshold else 0)
    return df


def transform_label_to_multi_classification(df):
    for label in list(df.columns):
        threshold_list = multi_classification_label_threshold[label]
        df[label] = df[label].apply(cal_val_by_multi_threshold, thresholds=threshold_list)
    return df


# 过采样平衡样本
def resample(df, label):
    columns_except_status = list(df.columns)
    columns_except_status.remove(label)

    x_data = df[columns_except_status]
    y_data = df[[label]]

    # sm = SMOTE(random_state=0)
    smt = SMOTETomek(random_state=0)
    x, y = smt.fit_resample(x_data, y_data)
    print("SMOTE过采样后，训练集的{}标签分布情况: {}".format(label, Counter(y[label])))

    df = pd.concat([x, y], axis=1)
    return df


# 做标准化、标签类型转换
def preprocess_data(df):
    df = df[feature_label_list]
    feature_df = preprocess_features(df[feature_list].copy())
    label_df = preprocess_labels(df[target_labels].copy())
    return pd.concat([feature_df, label_df], axis=1)


class MyDataset(data.Dataset):
    def __init__(self, df):
        self.x_df = df[feature_list]
        self.y_df = df[target_labels]

        self.x_data = np.array(self.x_df)
        self.y_data = np.array(self.y_df)

        self.x_data = self.x_data.astype(np.float32)
        # 使用CrossEntropyLoss()为损失函数注释掉下面这行
        self.y_data = self.y_data.astype(np.float32)

        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)


# 默认的label为status
def load_dataset():
    assert num_of_labels == len(label_types) == len(loss_functions_by_label),\
        "target_labels, label_types, loss_functions_by_label, 三个list的元素要一一对应"

    df = pd.read_csv(data_path)
    df = preprocess_data(df)

    train_df, test_df = train_test_split(df, test_size=0.4)
    train_df = resample(train_df, 'status')

    train_dataset = MyDataset(train_df)
    test_dataset = MyDataset(test_df)

    return train_dataset, test_dataset


def process_text(df):
    sentences = df['subject']
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_patter=r'\b\w+\b', min_df=2)
    return bigram_vectorizer.fit_transform(sentences).toarray()
