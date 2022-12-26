from Miner import *
import joblib
from SimpleParser import *
from tqdm import tqdm
from data_mining.Util import *

# changes created and closed within this time is selected by select_changes method.
before = {'Libreoffice': '2019', 'Eclipse': '2017', 'Gerrithub': '2019'}
after = {'Libreoffice': '2012', 'Eclipse': '2012', 'Gerrithub': '2016'}
# after = {'Libreoffice': '2018', 'Eclipse': '2016', 'Gerrithub': '2017'}

def main():
    # create data directories if mining for the first time
    initialize_dirs()

    # 1. Initialize miner with data root
    gerrit = Gerrit[project.lower()]
    if gerrit is None:
        print(f'Failed to get a valid gerrit type for {gerrit}. Exiting.')
        exit(-1)

    miner = Miner(gerrit=gerrit, root=root, replace=False)
    #
    # # 2. Download change details
    parameters = Parameters(
        status=Status.closed, start_index=0, end_index=-1, n_jobs=4, batch_size=100,
        after='', before='2019-00-00 00:00:00.000000000',
        fields=[Field.all_revisions, Field.all_files, Field.messages, Field.detailed_labels, Field.all_commits]
    )

    result = miner.change_details_mine(sub_directory=change_folder, parameters=parameters, timeout=300)
    for url, did_succeed in result:
        if did_succeed is False:
            print(f"{url} failed .")

    # 3. make a list of change_ids out of downloaded data
    make_change_list()

    # 4. make a list of accounts
    account_list_path = os.path.join(root, project + "_account_list.csv")
    make_account_list()

    # 5. mine accounts
    df = pd.read_csv(account_list_path)
    miner.profiles_mine(df['account_id'].values)

    # 6. download join dates
    account_ids = pd.read_csv(account_list_path)["account_id"].values
    miner.profiles_mine(sorted(account_ids))

    # 7. extract join date
    profile_root = f"{root}/profile"
    extract_join_dates(profile_root)

    # 8. select changes
    selected_changes_path = f"{root}/{project}_selected_changes.csv"
    select_changes(selected_changes_path)

    # 9. break batch change file contents into individual change file
    broken_changes_directory = f"{root}/changes"
    break_changes(broken_changes_directory)

    # this is better to do after selecting changes
    # 10. mine change diff using 'Mine file diff.py' file.

    # remove changes from the selected changes list for which file diff content was not found
    remove_changes_without_diff(selected_changes_path)


def is_profile_file(filename: str) -> bool:
    pattern = r'profile_[0-9]+.json'
    return bool(re.fullmatch(pattern, filename))


def extract_join_dates(profile_root):
    accounts = []
    joindates = []
    names = []
    filenames = [filename for filename in os.listdir(profile_root) if is_profile_file(filename)]
    for filename in filenames:
        filepath = os.path.join(profile_root, filename)
        # profile_json = json.load(open(filepath, "r"))
        with open(filepath, "r") as file:
            profile_json = json.load(file)

            profile = Profile(profile_json)
            accounts.append(profile.account_id)
            joindates.append(profile.registered_on)
            names.append(profile.name)

    joindate = pd.DataFrame({'account_id': accounts, 'name': names, 'registered_on': joindates})

    account_list = pd.read_csv(account_list_filepath)[['account_id']]
    account_list = account_list.merge(joindate, on=['account_id'], how='left', sort=True)

    accounts = account_list['account_id'].astype(int).values
    numerics = pd.to_numeric(pd.to_datetime(account_list['registered_on'])).values
    dates = pd.to_datetime(account_list['registered_on']).values

    # find the earliest record for this account in change list
    change_list_df = joblib.load(change_list_filepath)
    change_list_df['created'] = pd.to_datetime(change_list_df['created'])

    for index, account_id in tqdm(enumerate(accounts)):
        if dates[index] is not None:
            selected = change_list_df[change_list_df['created'] < dates[index]]
        else:
            selected = change_list_df

        # for (_, _, _, _, created, _, owner, reviewers, _, _) in selected.itertuples(name=None):
        for (_, _, _, _, created, _, _, owner, reviewers, _, _, _, _, _) in selected.itertuples(name=None):
            if account_id == owner or account_id in reviewers:
                # print(account_id, dates[index], change_dates[change_index])
                dates[index] = created
                break

    account_list['registered_on'] = dates

    null_index = account_list[account_list['registered_on'].isnull()].index.tolist()
    for index in null_index:
        # interpolate from prev and next dates
        prev_index = index - 1
        while prev_index > 0 and prev_index in null_index:
            prev_index -= 1

        next_index = index + 1
        while next_index < account_list.shape[0] and next_index in null_index:
            next_index += 1

        if prev_index >= 0 and next_index < account_list.shape[0]:
            diff_y = numerics[next_index] - numerics[prev_index]
            diff_x = accounts[next_index] - accounts[prev_index]
            y = numerics[prev_index] + (accounts[index] - accounts[prev_index]) * (diff_y / diff_x)
            dates[index] = pd.to_datetime(int(y))
        elif prev_index >= 0:
            dates[index] = dates[prev_index]
        elif next_index < account_list.shape[0]:
            dates[index] = dates[next_index]

    account_list['registered_on'] = dates
    account_list.to_csv(account_list_filepath, index=False)


def make_change_list():
    print("Making change list")
    filenames = [filename for filename in os.listdir(change_directory_path) if filename.endswith(".json")]
    change_details = {
        "change_id": [],
        "project": [],
        "subject": [],
        "created": [],
        "updated": [],
        "closed": [],
        "owner": [],
        "reviewers": [],
        "subsystems": [],
        "message_num": [],
        "revision_num": [],
        "duration": [],
        "status": []
    }

    for filename in tqdm(filenames):
        # print(f"Working on {filename}")
        with open(os.path.join(change_directory_path, filename), 'r', encoding='utf-8') as input_file:
            for change_json in load_change_jsons(input_file):

                for key in ["project", "created", "subject", "status"]:
                    change_details[key].append(change_json[key])

                change = Change(change_json)
                change_details["change_id"].append(change.change_number)
                change_details['owner'].append(change.owner)
                change_details['reviewers'].append(change.reviewers)
                change_details['subsystems'].append(change.subsystems)

                change_details['updated'].append(change.updated)
                change_details['closed'].append(change.closed)
                change_details['duration'].append(day_diff(change.closed, change.created))
                change_details['message_num'].append(len(change.messages))
                change_details['revision_num'].append(len(change.revisions))
                # break

    change_list_df = pd.DataFrame(change_details).sort_values(by=["change_id"])
    # need joblib because 'reviewers' column contains list of int and dataframe can't handle that
    change_list_df.drop_duplicates(['change_id'], inplace=True)
    joblib.dump(change_list_df, change_list_filepath)


def is_bot(name):
    name = name.lower()
    if (project.lower() in name) or name == 'do not use':
        return True

    words = name.split()
    for word in ['bot', 'chatbot', 'ci', 'jenkins']:
        if word in words:
            return True

    return False


def find_and_remove_bot_accounts():
    print("Finding and removing bot accounts")
    df = pd.read_csv(account_list_filepath)
    names = df['name'].fillna('').values
    account_ids = df['account_id'].values
    bots = []
    for index, name in enumerate(names):
        if is_bot(project.lower(), name):
            print(name)
            bots.append(account_ids[index])
    print()
    df = df[~df['account_id'].isin(bots)]
    df.to_csv(account_list_filepath, index=False)

    df = joblib.load(change_list_filepath)
    reviewers_list = df['reviewers'].values

    result = []
    for reviewer_list in reviewers_list:
        reviewers = []
        for account_id in reviewer_list:
            if account_id not in bots:
                reviewers.append(account_id)
        result.append(reviewers)
    df['reviewers'] = result
    joblib.dump(df, change_list_filepath)


def remove_changes_without_diff(selected_changes_path):
    change_numbers = [int(filename.split('_')[1]) for filename in os.listdir(diff_root)]
    selected_changes = pd.read_csv(selected_changes_path)
    old_number = selected_changes.shape[0]
    selected_changes = selected_changes[selected_changes['change_id'].isin(change_numbers)]

    print("Changes reduced from {0} to {1} after removing those without diff file.".format(
        old_number, selected_changes.shape[0]))
    selected_changes.to_csv(selected_changes_path, index=False)

    df = joblib.load(change_list_filepath)
    df = df[df['change_id'].isin(selected_changes['change_id'].values)]
    joblib.dump(df, selected_change_list_filepath)


def break_changes(broken_changes_directory):
    print("Breaking changes into individual files")
    if not os.path.exists(broken_changes_directory):
        os.mkdir(broken_changes_directory)

    filenames = [filename for filename in os.listdir(change_directory_path)]
    for filename in tqdm(filenames):
        with open(os.path.join(change_directory_path, filename), 'r', encoding='utf-8') as input_file:
            for change_json in load_change_jsons(input_file):
                change_number = change_json["_number"]
                with open(f'{broken_changes_directory}/{project}_{change_number}_change.json', 'w') as output_file:
                    json.dump(change_json, output_file, indent=4)


def make_account_list():
    change_list_df = joblib.load(change_list_filepath)
    accounts = set(change_list_df['owner'].values)
    accounts.update(set(np.concatenate(change_list_df['reviewers'].values)))

    accounts = pd.DataFrame(accounts, columns=['account_id'])
    accounts['account_id'] = accounts['account_id'].astype(int).values
    accounts.to_csv(account_list_filepath, index=False)


def is_owner_only_reviewer(owner, reviewers):
    return [owner] == reviewers


def select_changes(output_path):
    print('Selecting changes for project ' + project)

    # 1. Create selected changes list
    df = joblib.load(change_list_filepath)
    print(df.shape, df['created'].min(), df['created'].max())
    # filter by mining time
    df = df[(df['created'] >= after[project]) & (df['created'] <= before[project])]

    print(df.shape)
    df['subject'] = df['subject'].str.lower()
    for label in ["not merge", "ignore ", "wip ", "wip:", "[wip]", "work in progress"]:
        df = df[df['subject'].apply(lambda x: label not in x)]

    # select projects with at least 200 changes
    grouped = df.groupby(["project"]).size().reset_index(name='counts')
    grouped = grouped[grouped.counts >= 200]
    grouped.to_csv(f"{root}/{project}_selected_projects.csv", index=False)
    df = df[df.project.isin(grouped.project)]

    df = df[~df[['owner', 'reviewers']].apply(lambda x: is_owner_only_reviewer(x[0], x[1]), axis=1)]
    df[['project', 'change_id']].to_csv(output_path, index=False)
    print(df.shape)


if __name__ == '__main__':
    main()

# Selecting changes for project Libreoffice
# (61062, 9) 2012-03-06 10:46:41.000000000 2018-11-29 21:42:49.000000000
# (61062, 9)
# (56869, 9)
# Changes reduced from 56869 to 56241 after removing those without diff file.
# Selecting changes for project Gerrithub
# (61989, 9) 2014-03-12 16:16:06.000000000 2018-11-29 22:33:47.000000000
# (60192, 9)
# (33400, 9)
# Changes reduced from 33400 to 33020 after removing those without diff file.
# Selecting changes for project Eclipse
# (113427, 9) 2009-10-01 23:54:20.000000000 2018-11-29 19:23:00.000000000
# (68343, 9)
# (57811, 9)
# Changes reduced from 57811 to 57351 after removing those without diff file.
