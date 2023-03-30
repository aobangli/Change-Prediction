import torch
import torch.utils.data as data
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
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
            if label_type == TaskType.Binary_Classification:
                df[[label]] = transform_label_to_binary_classification(df[[label]].copy())
            elif label_type == TaskType.Multiple_Classification:
                df[[label]] = transform_label_to_multi_classification(df[[label]].copy())
            elif label_type == TaskType.Regression:
                if apply_minmax_to_regression:
                    # scaler = MinMaxScaler()
                    scaler = StandardScaler()
                    df[[label]] = scaler.fit_transform(df[[label]])
                    scalers_buffer[label] = scaler
    return df[target_labels]


def transform_label_to_binary_classification(df):
    for label in list(df.columns):
        threshold = binary_classification_label_threshold[label]
        df[label] = df[label].apply(lambda x: 1 if x > threshold else 0)
    return df


def transform_label_to_multi_classification(df):
    for label in list(df.columns):
        threshold_list = multi_classification_label_threshold[label]
        df[label] = df[label].apply(classify_by_multi_threshold, thresholds=threshold_list)
    return df


# 过采样平衡样本
def resample(df, label):
    columns_except_status = list(df.columns)
    columns_except_status.remove(label)

    x_data = df[columns_except_status]
    y_data = df[[label]]

    sm = SMOTE(random_state=0)
    # smt = SMOTETomek(random_state=0)
    x, y = sm.fit_resample(x_data, y_data)
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
    # reordered_feature_list和reordered_label_list可以指定特征以及特征列的顺序，有些模型对顺序有要求
    def __init__(self, df, reordered_feature_list=feature_list, reordered_label_list=target_labels):
        self.x_df = df[reordered_feature_list]
        self.y_df = df[reordered_label_list]

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


def prepare_dataframe():
    assert num_of_labels == len(label_types) == len(loss_functions_by_label), \
        "target_labels, label_types, loss_functions_by_label, 三个list的元素要一一对应"

    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    return df


def load_splited_dataframe():
    df = prepare_dataframe()
    train_df, test_df = train_test_split(df, test_size=0.4, shuffle=False)
    return train_df, test_df


# 默认的label为status
def load_dataset():
    train_df, test_df = load_splited_dataframe()
    # train_df = resample(train_df, 'status')

    train_dataset = MyDataset(train_df)
    test_dataset = MyDataset(test_df)

    return train_dataset, test_dataset


# 按时间序列划分训练集和测试集，每次将一个季度的数据加入训练集，将下一个季度的数据作为测试集
def load_by_period():
    df = pd.read_csv(data_path)

    df['created'] = pd.to_datetime(df['created'])
    groups = list(df.groupby(pd.Grouper(key='created', freq='Q')))
    df_slices = [item[1] for item in list(groups)]
    res = []

    for index, df_item in enumerate(df_slices):
        if index == 0:
            continue
        train_df = pd.concat(df_slices[0:index], axis=0)
        test_df = df_item
        res.append((preprocess_data(train_df.copy()), preprocess_data(test_df.copy())))
    return res


# 按时间序列划分训练集和测试集，将八个季度的数据作为训练集，下两个季度的数据作为测试集，以滑动窗口的形式依次往后滑动
def load_by_period_slide(train_size=8, test_size=2):
    df = pd.read_csv(data_path)

    df['created'] = pd.to_datetime(df['created'])
    groups = list(df.groupby(pd.Grouper(key='created', freq='Q')))
    df_slices = [item[1] for item in list(groups)]
    res = []

    for index, df_item in enumerate(df_slices):
        if index < train_size:
            continue
        if index + test_size > len(df_slices):
            break
        train_df = pd.concat(df_slices[index - train_size: index], axis=0)
        # test_df = df_item
        test_df = pd.concat(df_slices[index: index + test_size], axis=0)
        res.append((preprocess_data(train_df.copy()), preprocess_data(test_df.copy())))
    return res

def process_text(df):
    sentences = df['subject']
    # sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(sentences)
    print(embeddings.shape)
    return embeddings

