import torch.nn as nn
from config.TaskType import TaskType
from custom_loss_functions.losses import *

# 原始数据集路径
original_path = '../data/Eclipse.csv'
data_path = '../data/Eclipse_all_labels.csv'
result_output_path = '../data/output'

features_group = {
    'author': ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
               'author_merge_ratio_in_project', 'total_change_num', 'author_review_num'],
    'text': ['description_length', 'is_documentation', 'is_bug_fixing', 'is_feature'],
    'project': ['project_changes_per_week', 'project_merge_ratio', 'changes_per_author'],
    'reviewer': ['num_of_reviewers', 'num_of_bot_reviewers', 'avg_reviewer_experience', 'avg_reviewer_review_count'],
    'code': ['lines_added', 'lines_deleted', 'files_added', 'files_deleted', 'files_modified',
             'num_of_directory', 'modify_entropy', 'subsystem_num']
}


def get_initial_feature_list() -> [str]:
    features = []
    for group in features_group:
        features.extend(features_group[group])
    return features


# 部分模型需要区分稠密特征（数值型）
dense_features_cols = ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
                       'author_merge_ratio_in_project', 'total_change_num', 'author_review_num',
                       'description_length', 'project_changes_per_week', 'project_merge_ratio',
                       'changes_per_author',
                       'num_of_reviewers', 'num_of_bot_reviewers', 'avg_reviewer_experience',
                       'avg_reviewer_review_count',
                       'lines_added', 'lines_deleted', 'files_added', 'files_deleted', 'files_modified',
                       'num_of_directory', 'modify_entropy', 'subsystem_num']

# 稀疏特征（分类型）
sparse_features_cols = ['is_documentation', 'is_bug_fixing', 'is_feature']

# 每个稀疏特征的值的类别数，部分模型中需要指定，用于为每个稀疏特征构造embedding的参数
sparse_features_val_num = [2, 2, 2]

# 数值型特征个数
num_of_dense_feature = len(dense_features_cols)

# 原始数据集里的所有label
# all_labels = ['num_of_reviewers', 'rounds', 'time', 'avg_score', 'status']
# 原始数据为回归型的label
regression_labels = ['num_of_reviewers', 'rounds', 'time', 'avg_score']
# 需要预测的label
target_labels = ['rounds', 'time', 'avg_score', 'status']
# 要预测的label数量
num_of_labels = len(target_labels)
# 是否对回归任务label进行归一化
apply_minmax_to_regression = True
# 指定每种label的任务类型，与上面的labels一一对应
# label_types = [
#     TaskType.Regression,
#     TaskType.Regression,
#     TaskType.Regression,
#     TaskType.Binary_Classification
# ]
label_types = [
    TaskType.Multiple_Classification,
    TaskType.Multiple_Classification,
    TaskType.Multiple_Classification,
    TaskType.Binary_Classification
]
# 指定每个label的激活函数，与上面labels一一对应
# 注意：对于二分类任务，widedeep和deepcross模型使用BCELoss()，tabtransformer模型使用BCEWithLogitsLoss()
# loss_functions_by_label = [AsymmetricLoss(gamma_neg=1, gamma_pos=2, clip=0)]
# loss_functions_by_label = [nn.BCELoss()]
# loss_functions_by_label = [nn.BCEWithLogitsLoss()]
# loss_functions_by_label = [
#     nn.MSELoss(),
#     nn.MSELoss(),
#     nn.MSELoss(),
#     nn.BCELoss()
# ]
loss_functions_by_label = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.BCELoss()
]
# 将数值型标签转换为二分类时的阈值，大于阈值取1，反之取0
binary_classification_label_threshold = {'num_of_reviewers': 1, 'rounds': 2, 'time': 7, 'avg_score': 1}

# 将数值型标签转换为多分类的阈值区间，从0开始取值，除了两侧的区间外，每个区间左开右闭
multi_classification_label_threshold = \
    {'num_of_reviewers': [2, 4], 'rounds': [1, 3], 'time': [1, 7], 'avg_score': [1, 1.75]}


# 根据阈值区间，将给定数值型数据映射为多分类特征
def classify_by_multi_threshold(data, thresholds):
    for index, threshold in enumerate(thresholds):
        if data <= threshold:
            return index
    return len(thresholds)


# 根据任务类型，获取输出维度list，二分类任务与回归任务为1，多分类任务类别数
def get_task_out_dims():
    dims = []
    for index, task_name in enumerate(target_labels):
        task_type = label_types[index]
        if task_type == TaskType.Multiple_Classification:
            dim = len(multi_classification_label_threshold[task_name]) + 1
        else:
            dim = 1
        dims.append(dim)
    return dims


task_out_dims = get_task_out_dims()

# 所有特征列表
feature_list = dense_features_cols + sparse_features_cols
# 所有特征和标签列表
feature_label_list = feature_list + target_labels

# 对label做归一化时，保存对应的scaler，方便后面对模型预测结果还原
scalers_buffer = {}
