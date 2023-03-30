import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_loader import load_data
from config.TrainConfig import *


def run_binary_cls_task(label, train_df, test_df):

    x_train = train_df[feature_list]
    y_train = train_df[label]
    x_test = test_df[feature_list]
    y_test = test_df[label]

    rfc = RandomForestClassifier(class_weight='balanced', random_state=23)

    rfc.fit(x_train, y_train)

    preds = rfc.predict(x_test)
    score = rfc.predict_proba(x_test)[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(classification_report(y_test, preds))
        print("auc = ", roc_auc_score(y_test, score))

    # print("Features sorted by their score:")
    # print(sorted(zip(map(lambda val: round(val, 4), rfc.feature_importances_), feature_list), reverse=True))


def run_multi_cls_task(label, train_df, test_df):
    x_train = train_df[feature_list]
    y_train = train_df[label]
    x_test = test_df[feature_list]
    y_test = test_df[label]

    rfc = RandomForestClassifier(class_weight='balanced', random_state=23)

    rfc.fit(x_train, y_train)

    preds = rfc.predict(x_test)
    score = rfc.predict_proba(x_test)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print(classification_report(y_test, preds))
        print("auc = ", roc_auc_score(y_test, score, multi_class='ovr'))

    # print("Features sorted by their score:")
    # print(sorted(zip(map(lambda val: round(val, 4), rfc.feature_importances_), feature_list), reverse=True))


def run(train_df, test_df):
    for task_name, label_type in zip(target_labels, label_types):
        print(f'########## {task_name} ##########')
        if label_type == TaskType.Binary_Classification:
            run_binary_cls_task(task_name, train_df, test_df)
        elif label_type == TaskType.Multiple_Classification:
            run_multi_cls_task(task_name, train_df, test_df)
        else:
            raise TypeError("任务类型错误！")


if __name__ == "__main__":
    train_df, test_df = load_data.load_splited_dataframe()
    run(train_df, test_df)
    # df_list = load_data.load_by_period()
    # for round_index, (_train_df, _test_df) in tqdm(enumerate(df_list)):
    #     print(f'=============== run {round_index} round! ===============')
    #     print(f'train_size: {_train_df.shape[0]}    test_size: {_test_df.shape[0]}')
    #     run(_train_df, _test_df)



