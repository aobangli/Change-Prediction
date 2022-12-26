import pandas as pd
import os, csv
import re, json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import numpy as np
from Config import *


def run_model(model, df, feature_list=initial_feature_list):
    total_result = None
    for run in range(runs):
        result = Result()
        for fold in range(1, folds):
            train_size = df.shape[0] * fold // folds
            test_size = min(df.shape[0] * (fold + 1) // folds, df.shape[0])

            x_train, y_train = df.loc[:train_size - 1, feature_list], \
                               df.loc[:train_size - 1, target]
            x_test, y_test = df.loc[train_size:test_size - 1, feature_list], \
                             df.loc[train_size:test_size - 1, target]

            clf =model
            clf.fit(x_train, y_train)

            y_prob = clf.predict_proba(x_test)[:, 1]
            result.calculate_result(y_test, y_prob, fold, False)

        result_df = result.get_df()
        if run:
            total_result += result_df
        else:
            total_result = result_df

    total_result /= runs
    total_result = total_result.mean()
    print(total_result[['auc', 'f1_score_m', 'f1_score_a']].values)
    print()

class Result:
    def __init__(self):
        self.folds = []
        self.auc = []
        self.accuracy = []
        self.effectiveness = []

        self.precision_m = []
        self.recall_m = []
        self.f1_score_m = []

        self.precision_a = []
        self.recall_a = []
        self.f1_score_a = []

    # evaluates the percentage of merged code changes over the top K%
    # suspicious merged code changes
    @staticmethod
    def cost_effectiveness(y_true, y_score, k):
        df = pd.DataFrame({'class': y_true, 'pred': y_score})
        df = df.sort_values(by=['pred'], ascending=False).reset_index(drop=True)
        if k > 100:
            print('K must be  > 0 and < 100')
            return -1
        df = df.iloc[:df.shape[0] * k // 100]

        merged_changes = df[df['class'] == 1].shape[0]
        changes = df.shape[0]

        if changes:
            return merged_changes / changes
        else:
            return 0

    def calculate_result(self, y_true, y_score, fold=None, verbose=False):
        if fold is not None:
            self.folds.append(fold)

        auc = roc_auc_score(y_true, y_score)
        cost_effectiveness = Result.cost_effectiveness(y_true, y_score, 20)
        if verbose: print(f'AUC {auc}, cost effectiveness {cost_effectiveness}.')
        self.auc.append(auc)

        self.effectiveness.append(cost_effectiveness)

        y_pred = np.round(y_score)

        self.accuracy.append(accuracy_score(y_true, y_pred))

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None)
        if verbose: print(f'precision {precision}, recall {recall}, f1_score {f1_score}.')
        self.precision_a.append(precision[0])
        self.recall_a.append(recall[0])
        self.f1_score_a.append(f1_score[0])

        self.precision_m.append(precision[1])
        self.recall_m.append(recall[1])
        self.f1_score_m.append(f1_score[1])

    def get_df(self):
        return pd.DataFrame({
            'fold': self.folds,
            'auc': self.auc,
            'accuracy': self.accuracy,
            'cost_effectiveness': self.effectiveness,
            'precision_m': self.precision_m,
            'recall_m': self.recall_m,
            'f1_score_m': self.f1_score_m,
            'precision_a': self.precision_a,
            'recall_a': self.recall_a,
            'f1_score_a': self.f1_score_a,
        })

    def process(self, number):
        return np.round(np.mean(number), 2)

    def show(self):
        print(
            f"{self.process(self.auc)} & {self.process(self.effectiveness)} & {self.process(self.f1_score_m)} & {self.process(self.precision_m)} & {self.process(self.recall_m)} & {self.process(self.f1_score_a)} & {self.process(self.precision_a)} & {self.process(self.recall_a)}")


def initialize_dirs():
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    for project in projects:
        if not os.path.exists(f'{data_folder}/{project}'):
            os.mkdir(f'{data_folder}/{project}')

        if not os.path.exists(f'{result_folder}/{project}'):
            os.mkdir(f'{result_folder}/{project}')

        for dirname in ['change', 'changes', 'diff', 'profile']:
            if not os.path.exists(f'{data_folder}/{project}/{dirname}'):
                os.mkdir(f'{data_folder}/{project}/{dirname}')


def initialize(path, file_header):
    with open(path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', dialect='excel')
        writer.writerow(file_header)
        csvfile.close()


def day_diff(date1, date2):
    if type(date1) == str:
        date1 = pd.to_datetime(date1)
    if type(date2) == str:
        date2 = pd.to_datetime(date2)

    diff = date1 - date2
    return diff.days + diff.seconds / 86400.0


def is_bot(project, name):
    project = project.lower()
    name = name.lower()
    if project in name or name == 'do not use':
        return True

    words = name.split()
    for word in ['bot', 'chatbot', 'ci', 'jenkins']:
        if word in words:
            return True

    return False


def safe_drop_column(df, columns):
    for col in columns:
        if col in df.columns:
            df.drop([col], axis=1, inplace=True)
        else:
            print("Error: column {0} not found".format(col))
    return df


def is_change_file(filename: str) -> bool:
    status = ["open", "closed", "merged", "abandoned"]
    for s in status:
        if s in filename:
            return True
    return False


def is_profile_file(filename: str) -> bool:
    pattern = r'profile_[0-9]+.json'
    return bool(re.fullmatch(pattern, filename))


def is_profile_details_file(filename: str) -> bool:
    pattern = r'profile_details_[0-9]+.json'
    return bool(re.fullmatch(pattern, filename))


def make_date(date):
    # date = re.sub(r"\.[0-9]+", "", date)
    # return datetime.datetime.strptime(date, format='%Y-%m-%d %H:%M:%S')
    return pd.to_datetime(date)


'''
these jsons are generally in format list of jsons
but sometimes they are list of one list of jsons
'''
def load_change_jsons(input_file):
    change_json = json.load(input_file)
    while type(change_json) == list and len(change_json) != 0 and type(change_json[0]) == list:
        change_json = change_json[0]
    return change_json


def toJSON ( object ) :
    return json.dumps(object , default=lambda o : o.__dict__ , sort_keys=True , indent=4)

def subsystem_of(file_path):
    str_list = file_path.split('/')
    if len(str_list) == 1:
        return ''
    else:
        if str_list[0] == '':
            return str_list[1]
        return str_list[0]


def directory_of(file_path):
    return os.path.dirname(file_path)


def is_nonhuman(author_name):
    return 'CI' in author_name.split(' ') or \
        author_name == 'jenkins' or \
        author_name == 'Jenkins' or \
        author_name == 'Eclipse Genie'
        # try:
        #     reviewers.remove(author_id)
        #     project_set_instance.non_natural_human.add(author_id)
        # except KeyError:
        #     pass