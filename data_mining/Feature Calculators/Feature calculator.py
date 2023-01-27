import joblib
from tqdm import tqdm

import config.TrainConfig
from data_mining.Util import *
from data_mining.Miners.SimpleParser import *
import pandas as pd
from datetime import timedelta
import numpy as np
from config.DataMiningConfig import *

account_list_df = pd.read_csv(account_list_filepath)
account_list_df['registered_on'] = pd.to_datetime(account_list_df['registered_on'])
account_list_df['name'] = account_list_df['name'].apply(str)

change_list_df = joblib.load(selected_change_list_filepath)
change_list_df = change_list_df.sort_values(by=['change_id']).reset_index(drop=True)
for col in ['created', 'updated', 'closed']:
    change_list_df.loc[:, col] = change_list_df[col].apply(pd.to_datetime)

lookback = 60
default_changes = 1
default_merge_ratio = 0.5


def main():
    features_list = initial_feature_list
    file_header = ["project", "change_id", 'created', 'subject'] + features_list + ['status', 'time', 'rounds']
    output_file_name = f"{root}/{project}.csv"

    initialize(output_file_name, file_header)
    csv_file = open(output_file_name, "a", newline='', encoding='utf-8')
    file_writer = csv.writer(csv_file, dialect='excel')

    change_numbers = change_list_df['change_id'].values

    # it is important to calculate in sorted order of created.
    # Change numbers are given in increasing order of creation time
    count = 0
    # for change_number in change_numbers:
    for change_number in tqdm(change_numbers):
        # print(change_number)

        filename = f'{project}_{change_number}_change.json'
        filepath = os.path.join(changes_root, filename)
        if not os.path.exists(filepath):
            print(f'{filename} does not exist')
            continue

        change = Change(json.load(open(filepath, "r")))
        if not change.is_real_change():
            continue

        current_date = pd.to_datetime(change.first_revision.created)
        calculator = FeatureCalculator(change, current_date)

        author_features = calculator.author_features
        reviewer_features = calculator.reviewer_features
        file_features = calculator.file_features
        # file_diff_features = calculator.diff_features

        project_features = calculator.project_features
        description_features = calculator.description_features

        status = 1 if change.status == 'MERGED' else 0
        feature_vector = [
            change.project, change.change_number, change.created, change.subject,

            author_features['author_experience'], author_features['author_merge_ratio'],
            author_features['author_changes_per_week'], author_features['author_merge_ratio_in_project'],
            author_features['total_change_num'], author_features['author_review_num'],

            description_features['description_length'], description_features['is_documentation'],
            description_features['is_bug_fixing'], description_features['is_feature'],

            project_features['project_changes_per_week'], project_features['project_merge_ratio'],
            project_features['changes_per_author'],

            reviewer_features['num_of_reviewers'], reviewer_features['num_of_bot_reviewers'],
            reviewer_features['avg_reviewer_experience'],
            reviewer_features['avg_reviewer_review_count'],

            file_features['lines_added'], file_features['lines_deleted'], # file_diff_features['lines_updated'],
            file_features['modify_entropy'],

            file_features['files_added'], file_features['files_deleted'], file_features['files_modified'],
            file_features['num_of_directory'], file_features['subsystem_num'],

            status,
            day_diff(change.closed, change.created),
            len(change.revisions)
        ]
        file_writer.writerow(feature_vector)

        count += 1
        if count % 100 == 0:
            csv_file.flush()
            # break

    csv_file.close()

    features = pd.read_csv(output_file_name)
    features.drop_duplicates(['change_id'], inplace=True)
    features.sort_values(by=['change_id']).to_csv(output_file_name, index=False, float_format='%.2f')


class FeatureCalculator:
    def __init__(self, change, current_date):
        self.change = change
        self.project = change.project
        self.current_date = current_date
        self.old_date = current_date - timedelta(days=lookback)

    @property
    def author_features(self):
        features, owner = {}, self.change.owner
        registered_on = account_list_df[account_list_df['account_id'] == owner]['registered_on'].values
        authors_work = change_list_df[
            (change_list_df['owner'] == owner) & (change_list_df['created'] < self.current_date)]
        features['total_change_num'] = authors_work.shape[0]
        # first_date = authors_work['created'].min()
        first_date = authors_work['created'].min() if authors_work.shape[0] > 0 else self.old_date

        if len(registered_on) == 0 or registered_on[0] > self.current_date:
            features['author_experience'] = max(0, day_diff(self.current_date, first_date) / 365.0)
        else:
            features['author_experience'] = day_diff(self.current_date, registered_on[0]) / 365.0

        ongoing_works = change_list_df[(self.old_date <= change_list_df['updated'])
                                       & (change_list_df['created'] <= self.current_date)]

        features['author_review_num'] = ongoing_works[ongoing_works['reviewers'].apply(lambda x: owner in x)].shape[0]

        finished_works = ongoing_works[(ongoing_works['owner'] == owner) & (ongoing_works['updated'] <= self.current_date)]
        merged_works = finished_works[finished_works['status'] == 'MERGED']

        if finished_works.shape[0] >= default_changes:
            features['author_merge_ratio'] = float(merged_works.shape[0]) / finished_works.shape[0]
        else:
            features['author_merge_ratio'] = default_merge_ratio

        weeks = max(day_diff(self.current_date, max(first_date, self.old_date)) / 7.0, 1)
        features['author_changes_per_week'] = finished_works.shape[0] / weeks
        if np.isnan(features['author_changes_per_week']):
            print(finished_works.shape[0])
            print(weeks)

        finished_changes_in_project = finished_works[finished_works['project'] == self.project].shape[0]
        if finished_changes_in_project >= default_changes:
            features['author_merge_ratio_in_project'] = float(
                merged_works[merged_works['project'] == self.project].shape[0]) / finished_changes_in_project
        else:
            features['author_merge_ratio_in_project'] = default_merge_ratio
        return features

    @property
    def project_features(self):
        features = {}

        works = change_list_df[change_list_df['project'] == self.project]
        finished_works = works[(self.old_date <= works['updated']) & (works['updated'] <= self.current_date)]

        if finished_works.shape[0] >= default_changes:
            features['project_merge_ratio'] = float(finished_works[finished_works['status'] == 'MERGED'].shape[0]) / \
                                              finished_works.shape[0]
        else:
            features['project_merge_ratio'] = default_merge_ratio

        # per week changes in the last lookback days
        features['project_changes_per_week'] = finished_works.shape[0] * 7.0 / lookback

        owners = finished_works['owner'].nunique()
        features['changes_per_author'] = 0
        if owners:
            features['changes_per_author'] = float(finished_works.shape[0]) / owners

        return features

    @property
    def reviewer_features(self):
        features, reviewer_list = {}, self.change.reviewers
        ongoing_works = change_list_df[(self.old_date <= change_list_df['updated'])
                                      & (change_list_df['created'] <= self.current_date)]

        avg_experience = avg_num_review = 0.0
        count = bot = 0
        for reviewer_id in reviewer_list:
            result = account_list_df[account_list_df['account_id'] == reviewer_id]
            registered_on = result['registered_on'].values

            if len(registered_on) == 0 or self.current_date < registered_on[0]:
                continue

            if is_bot(self.project, result['name'].values[0]):
                bot += 1
                continue
            work_experience = day_diff(self.current_date, registered_on[0]) / 365.0  # convert to year
            avg_experience += work_experience

            avg_num_review += ongoing_works[ongoing_works['reviewers'].apply(lambda x: reviewer_id in x)].shape[0]
            count += 1

        if count:
            avg_experience /= count
            avg_num_review /= count

        features['num_of_reviewers'] = count
        features['num_of_bot_reviewers'] = bot
        features['avg_reviewer_experience'] = avg_experience
        features['avg_reviewer_review_count'] = avg_num_review
        return features

    @property
    def file_features(self):
        features, files = {}, self.change.files

        files_added = files_deleted = 0
        lines_added = lines_deleted = 0

        directories = set()
        subsystems = set()
        for file in files:
            lines_added += file.lines_inserted
            lines_deleted += file.lines_deleted

            if file.status == 'D': files_deleted += 1
            if file.status == 'A': files_added += 1

            names = config.TrainConfig.path.split('/')
            if len(names) > 1:
                directories.update([names[-2]])
                subsystems.update(names[0])

        lines_changed = lines_added + lines_deleted

        features['lines_added'] = lines_added
        features['lines_deleted'] = lines_deleted

        features['num_of_directory'] = len(directories)
        features['subsystem_num'] = len(subsystems)

        features['files_added'] = files_added
        features['files_deleted'] = files_deleted
        features['files_modified'] = len(files) - files_deleted - files_added

        # Entropy is defined as: −Sum(k=1 to n)(pk∗log2pk). Note that n is number of files
        # modified in the change, and pk is calculated as the proportion of lines modified in file k among
        # lines modified in this code change.
        modify_entropy = 0
        if lines_changed:
            for file in files:
                lines_changed_in_file = file.lines_deleted + file.lines_inserted
                if lines_changed_in_file:
                    pk = float(lines_changed_in_file) / lines_changed
                    modify_entropy -= pk * np.log2(pk)

        features['modify_entropy'] = modify_entropy
        return features

    @property
    def description_features(self):
        subject = self.change.subject.lower()
        features = {'is_documentation': False, 'is_bug_fixing': False, 'is_feature': False,
                    'description_length': len(subject.split())}

        for word in ['fix', 'bug', 'defect']:
            if word in subject:
                features['is_bug_fixing'] = True
                return features
        for word in ['doc', 'copyright', 'license']:
            if word in subject:
                features['is_documentation'] = True
                return features

        features['is_feature'] = True
        return features

    # @property
    # def diff_features(self):
    #     filepath = os.path.join(diff_root, f"{project}_{self.change.change_number}_diff.json")
    #     diff_json = json.load(open(filepath, 'r'))
    #
    #     segs_added = segs_deleted = segs_updated = lines_added = lines_deleted = lines_updated = modify_entropy = 0
    #
    #     try:
    #         files = list(diff_json.values())[0].values()
    #         for file in files:
    #             for content in file['content']:
    #                 change_type = list(content.keys())
    #                 if change_type == ['a']:
    #                     segs_deleted += 1
    #                     lines_deleted += len(content['a'])
    #                 elif change_type == ['a', 'b']:
    #                     segs_updated += 1
    #                     lines_updated += len(content['a'])
    #                 elif change_type == ['b']:
    #                     segs_added += 1
    #                     lines_added += len(content['b'])
    #
    #         lines_changed = lines_added + lines_deleted + lines_updated
    #         if lines_changed:
    #             for file in files:
    #                 lines_changed_in_file = 0
    #                 for content in file['content']:
    #                     change_type = list(content.keys())
    #                     if change_type == ['a']:
    #                         lines_changed_in_file += len(content['a'])
    #                     elif change_type == ['a', 'b']:
    #                         lines_changed_in_file += len(content['a'])
    #                     elif change_type == ['b']:
    #                         lines_changed_in_file += len(content['b'])
    #
    #                 if lines_changed_in_file:
    #                     pk = float(lines_changed_in_file) / lines_changed
    #                     modify_entropy -= pk * np.log2(pk)
    #
    #     except IndexError:
    #         print('Error for {0}'.format(self.change.change_number))
    #
    #     return {
    #         'segs_added': segs_added,
    #         'segs_updated': segs_updated,
    #         'segs_deleted': segs_deleted,
    #         'lines_added': lines_added,
    #         'lines_deleted': lines_deleted,
    #         'lines_updated': lines_updated,
    #         'modify_entropy': modify_entropy
    #     }


if __name__ == '__main__':
    main()