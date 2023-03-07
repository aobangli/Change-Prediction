from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data_loader import load_data
from config.TrainConfig import *


def run_binary_cls_task(label):
    y = df[label]

    x = df[feature_list]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    rfc = RandomForestClassifier(class_weight='balanced', random_state=23)

    rfc.fit(x_train, y_train)

    preds = rfc.predict(x_test)
    score = rfc.predict_proba(x_test)[:, 1]

    print(classification_report(y_test, preds))
    print("auc = ", roc_auc_score(y_test, score))

    print("Features sorted by their score:")
    print(sorted(zip(map(lambda val: round(val, 4), rfc.feature_importances_), feature_list), reverse=True))



def run_multi_cls_task(label):
    y = df[label]

    x = df[feature_list]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    rfc = RandomForestClassifier(class_weight='balanced', random_state=23)

    rfc.fit(x_train, y_train)

    preds = rfc.predict(x_test)
    score = rfc.predict_proba(x_test)

    print(classification_report(y_test, preds))
    print("auc = ", roc_auc_score(y_test, score, multi_class='ovr'))

    print("Features sorted by their score:")
    print(sorted(zip(map(lambda val: round(val, 4), rfc.feature_importances_), feature_list), reverse=True))


if __name__ == "__main__":
    df = load_data.prepare_dataframe()

    for task_name, label_type in zip(target_labels, label_types):
        print(f'########## {task_name} ##########')
        if label_type == TaskType.Binary_Classification:
            run_binary_cls_task(task_name)
        elif label_type == TaskType.Multiple_Classification:
            run_multi_cls_task(task_name)
        else:
            raise TypeError("任务类型错误！")
