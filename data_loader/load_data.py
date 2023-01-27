import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from config.TrainConfig import *

feature_list = get_initial_feature_list()

all_feature_list = list(feature_list)
all_feature_list.extend(target_labels)


def preprocess_features(df):
    sparse_columns = [column for column in df.columns if column not in dense_features_cols]
    columns = dense_features_cols + sparse_columns

    # 进行编码  类别特征编码
    for feature in sparse_columns:
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
    return df[columns]


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


class MyDataset(data.Dataset):
    def __init__(self, target):
        self.path = path
        self.df = pd.read_csv(self.path)

        self.df = self.df[all_feature_list]

        self.x_df = preprocess_features(self.df[feature_list].copy())

        for label, label_type in zip(target_labels, label_types):
            if label in regression_labels:
                if label_type == LabelType.Binary_Classification:
                    self.df[[label]] = transform_label_to_binary_classification(self.df[[label]].copy())
                elif label_type == LabelType.Multiple_Classification:
                    self.df[[label]] = transform_label_to_multi_classification(self.df[[label]].copy())
                elif label_type == LabelType.Regression:
                    # 暂不做归一化
                    # self.df[[label]] = MinMaxScaler().fit_transform(self.df[[label]])
                    self.df[[label]] = StandardScaler().fit_transform(self.df[[label]])

        self.y_df = self.df[target]

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
def load_dataset(label="status"):
    dataset = MyDataset(label)

    train_size = int(len(dataset) * 0.6)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def process_text(df):
    sentences = df['subject']
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_patter=r'\b\w+\b', min_df=2)
    return bigram_vectorizer.fit_transform(sentences).toarray()
