import os, csv, json, joblib
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from SimpleParser import Change
from Config import *

def main():
    miner = DiffMiner(project, replace=False, verbose=False)
    change_revision_file_ids_filename = os.path.join(f'{miner.root}/{project}_change_revision_file_ids.csv')
    # create_change_summary(miner.root, project, change_revision_file_ids_filename)

    # uncomment this if you need to rerun for mining rest of the file diffs.
    # list_not_mined_selected_changes()

    # only needed for the first time
    # change_revision_file_ids_filename = os.path.join(f'{miner.root}/{project}_change_revision_file_ids_left.csv')
    miner.mine(change_revision_file_ids_filename)


def create_change_summary(root, project, change_revision_file_ids_filename):
    change_root = f"{root}/change"

    selected_changes = pd.read_csv(f"{root}/{project}_selected_changes.csv")['change_id'].values

    filenames = [filename for filename in os.listdir(change_root) if is_change_file(filename)]
    output_file = open(change_revision_file_ids_filename, "w", newline='', encoding='utf-8')
    writer = csv.writer(output_file, delimiter=',')
    writer.writerow(['change_number', 'id', 'revision_id', 'file_id'])

    for filename in tqdm(filenames):
        filepath = os.path.join(change_root, filename)
        change_jsons = load_change_jsons(open(filepath, "r"))

        for change_json in change_jsons:
            change = Change(change_json)
            if change.change_number not in selected_changes:
                continue

            first_revision = change.first_revision
            if first_revision is None:
                continue
            if len(first_revision.files) > 1000:
                print(f'{change.change_number} has {len(first_revision.files)}  files !')
                continue
            for file in first_revision.files:
                writer.writerow([change.change_number, change.id, first_revision.id, file.path])

    # Libreoffice 7 changes without revisions
    # Eclipse 12 changes without revisions among 64844.
    output_file.close()


def is_change_file(filename: str) -> bool:
    status = ["open", "closed", "merged", "abandoned"]
    for s in status:
        if s in filename:
            return True
    return False


def load_change_jsons(input_file):
    change_json = json.load(input_file)
    while type(change_json) == list and len(change_json) != 0 and type(change_json[0]) == list:
        change_json = change_json[0]
    return change_json


def list_not_mined_selected_changes():
    filenames = os.listdir(diff_root)
    changes = [int(filename.split('_')[1]) for filename in filenames]
    df = pd.read_csv(f'{root}/{project}_selected_changes.csv')
    df = df[~df['change_id'].isin(changes)]
    # df.to_csv(f'{root}/{project}_selected_changes_left.csv', index=False)

    changes_left = df['change_id'].values

    df = pd.read_csv(f'{root}/{project}_change_revision_file_ids.csv', encoding='utf-8')
    df = df[df['change_number'].isin(changes_left)]
    df.to_csv(f'{root}/{project}_change_revision_file_ids_left.csv', encoding='utf-8', index=False)
    print(len(changes_left))


class DiffMiner:
    roots = {
        'eclipse': "https://git.eclipse.org/r",
        'libreoffice': "https://gerrit.libreoffice.org",
        'gerrithub': "https://review.gerrithub.io",
    }

    def __init__(self, project, replace=False, verbose=False):
        self.project = project
        self.replace = replace
        self.root = f"../../Data/{project}"
        self.diff_root = f"{self.root}/diff"
        self.verbose = verbose

        try:
            self.root_url = self.roots[project.lower()]
        except:
            print(project + " is not in roots dictionary")
            exit(-1)
        self.changes = {}

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        if not os.path.exists(self.diff_root):
            os.mkdir(self.diff_root)

    def download_diff(self, url, change_number, revision_id, file_id):
        if self.verbose: print(self.changes[change_number]['total'], url)

        response = requests.get(url, timeout=60)
        if response.status_code != 200:
            print("Response error. Status code {0}, change {1}".format(response.status_code, change_number))
            return

        data = response.text[4:]
        if data is None or len(data) == 0:
            if self.verbose: print("None or Empty : " + url)
            return

        try:
            self.changes[change_number][revision_id][file_id] = json.loads(data)
        except:
            print('invalid json format.')
            return

        self.changes[change_number]['total'] -= 1

        if self.changes[change_number]['total'] == 0:
            # print("Dumping {0}".format(change_number))
            del self.changes[change_number]['total']
            self.dump(change_number)

    def dump(self, change_number):
        path = f"{self.diff_root}/{self.project}_{change_number}_diff.json"
        try:
            f = open(path, "w")
            json.dump(self.changes[change_number], f, indent=4)
            f.close()
        except OSError as e:
            if self.verbose:
                print(f"Data could not be dumped to {path}.")
                print(e)

    def mine(self, change_revision_file_ids_filename):
        changes = {}
        # https://stackoverflow.com/questions/45529507/unicodedecodeerror-utf-8-codec-cant-decode-byte-0x96-in-position-35-invalid
        df = pd.read_csv(change_revision_file_ids_filename, encoding='utf-8')

        selected_changes = pd.read_csv(f'{self.root}/{self.project}_selected_changes.csv')['change_id'].values
        df = df[df['change_number'].isin(selected_changes)]

        change_numbers = df['change_number'].unique()
        print(df.shape, len(change_numbers))
        # change_numbers = change_numbers[:10]
        df = df[df['change_number'].isin(change_numbers)]

        for change_number in tqdm(change_numbers):
            changes[change_number] = {}
            files = df[df['change_number'] == change_number].reset_index(drop=True)
            changes[change_number][files.loc[0, 'revision_id']] = {}
            changes[change_number]['total'] = files.shape[0]

        self.changes = changes
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_url = {}
            for (_, change_number, id, revision_id, file_id) in tqdm(df.itertuples(name=None)):
                filename = f"{self.diff_root}/{self.project}_{change_number}_diff.json"
                if os.path.exists(filename) and not self.replace:
                    if self.verbose: print('Not replacing ' + filename)
                    continue

                url = f'{self.root_url}/changes/{id}/revisions/{revision_id}/files/{file_id.replace("/", "%2F")}/diff'
                future = executor.submit(self.download_diff, url, change_number, revision_id, file_id)
                future_to_url[future] = url

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    did_succeed = future.result()
                except Exception as exc:
                    print(f"{url} generated an exception: {exc}")
                # else:
                #     print(url + ' did succeed')


if __name__ == '__main__':
    main()
