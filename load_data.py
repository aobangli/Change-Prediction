import torch
from torch import nn
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

features_group = {
    'author': ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
               'author_merge_ratio_in_project', 'total_change_num', 'author_review_num'],
    'text': ['description_length', 'is_documentation', 'is_bug_fixing', 'is_feature'],
    'project': ['project_changes_per_week', 'project_merge_ratio', 'changes_per_author'],
    'reviewer': ['num_of_reviewers', 'num_of_bot_reviewers', 'avg_reviewer_experience', 'avg_reviewer_review_count'],
    'code': ['lines_added', 'lines_deleted', 'files_added', 'files_deleted', 'files_modified',
             'num_of_directory', 'modify_entropy', 'subsystem_num']
}
target = 'status'
path = 'data/Libreoffice.csv'


def get_initial_feature_list() -> [str]:
    features = []
    for group in features_group:
        features.extend(features_group[group])
    return features


feature_list = get_initial_feature_list()

all_feature_list = feature_list
all_feature_list.append(target)


class MyDataset(data.Dataset):
    def __init__(self):
        self.path = path
        self.df = pd.read_csv(self.path)

        self.df = self.df[all_feature_list]

        self.x_df = StandardScaler().fit_transform(self.df[feature_list])
        self.y_df = self.df[target]

        self.x_data = np.array(self.x_df)
        self.y_data = np.array(self.y_df)

        self.x_data = self.x_data.astype(np.float32)
        # self.y_data = self.y_data.astype(np.float32)

        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.y_data)


dataset = MyDataset()

train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


def load_dataset():
    return train_dataset, test_dataset

