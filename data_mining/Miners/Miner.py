import requests
import json, os
from concurrent.futures import as_completed, ThreadPoolExecutor
from enum import Enum


class Gerrit(Enum):
    android = "https://android-review.googlesource.com"
    chromium = "https://chromium-review.googlesource.com"
    cloudera = "https://gerrit.cloudera.org"
    eclipse = "https://git.eclipse.org/r"
    gerrithub = "https://review.gerrithub.io"
    go = "https://go-review.googlesource.com"
    libreoffice = "https://gerrit.libreoffice.org"
    opencord = "https://gerrit.opencord.org"
    openstack = "https://review.opendev.org"
    unlegacy = "https://gerrit.unlegacy-android.org"
    qt = "https://codereview.qt-project.org"

    def __str__(self):
        return f"{self.name}"


class Field(Enum):
    all_commits, all_revisions, all_files, change_actions, check, commit_footers, \
    current_actions, current_commit, current_files, \
    current_revision, detailed_accounts, detailed_labels, \
    download_commands, labels, messages, \
    no_limits, push_certificates, reviewed, reviewer_updates, \
    skip_mergeable, skip_diffstat, submittable, \
    tracking_ids, web_links = range(24)

    def __str__(self):
        return f"{self.name}"


class Status(Enum):
    abandoned, closed, merged, new, open, pending = range(6)

    def __str__(self):
        return f"{self.name}"


class Parameters:
    def __init__(self,
                 status: Status = Status.closed,
                 start_index: int = 0,
                 end_index: int = None,
                 after: str = '2019-01-01 00:00:00.000000000',
                 before: str = '2020-01-01 00:00:00.000000000',
                 fields: [Field] = None,
                 n_jobs: int = None,
                 batch_size: int = 100):
        """

        :param status: status of changes to download
        :param start_index: can be supplied to skip a number of changes from the list
        :param end_index: if -1 keeps downloading as much as possible
        :param after: timestamp
        :param before: timestamp
        :param fields: extra fields to download
        :param n_jobs: to parallelize to mining process
        :param batch_size: used to limit the returned results.
        """
        self.status = status
        self.start_index = start_index
        self.end_index = end_index
        self.after = after
        self.before = before

        if fields is None:
            self.fields = []
        else:
            self.fields = fields
        self.n_jobs = n_jobs
        self.batch_size = batch_size


class Miner:
    has_more_changes: bool = True

    def __init__(self, gerrit: Gerrit, root: str = None, replace: bool = False):
        self.gerrit = gerrit
        self.replace = replace
        if root is None:
            self.root = f"{self.gerrit}"
        else:
            self.root = root

        if not os.path.isdir(self.root):
            os.mkdir(self.root)

    def create_change_details_url(self, start_index: int, parameters: Parameters):
        url = f"{self.gerrit.value}/changes"
        # add query
        # Clients are allowed to specify more than one query by setting the q parameter multiple times.
        # In that case the result is an array of arrays, one per query in the same order the queries were given in.
        # however here we have written code for one query only
        url += "/?q="
        if parameters.status != '':
            url += f"status:{parameters.status}"

        if len(parameters.after) > 0:
            url += f"+after:\"{parameters.after}\""
        if len(parameters.before) > 0:
            url += f"+before:\"{parameters.before}\""

        # optional fields
        for field in parameters.fields:
            if type(field) is Field:
                url += f"&o={field}"
            else:
                print(f"Error: unknown field {field}")

        # The S or start query parameter can be supplied to skip a number of changes from the list
        url += f"&S={start_index}"

        if parameters.end_index is not None and parameters.end_index != -1:
            url += f"&n={min(parameters.batch_size, parameters.end_index - start_index)}"
        else:
            url += f"&n={parameters.batch_size}"

        # an exceptional case
        url = url.replace("NO_LIMIT", "NO-LIMIT").replace('=+', '=')

        return url

    def create_change_filename(self, status: Status, current_index: int, batch_size: int):
        return f"{self.gerrit}_{status}_{current_index}_{current_index + batch_size}.json"

    @staticmethod
    def parse(data: str):
        json_data = None
        try:
            json_data = json.loads(data)
        except ValueError:
            print(f"Invalid json. Response: {data}")
        return json_data

    def dump(self, path: str, data: json):
        if data is None:
            print("Error: data is None")
            return False
        if not os.path.exists(path) or self.replace:
            try:
                f = open(path, "w")
                json.dump(data, f, indent=4)
                f.close()
            except OSError as e:
                print(f"Data could not be dumped to {path}.")
                print(e)
                return False
        return True

    def download(self, url: str, timeout: int, filename: str, is_a_change: bool = False) -> bool:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            print("Status code {0} for {1}".format(response.status_code, url))
            return False

        data = response.text[4:]
        if len(data) == 0:
            print(f"Error: {url} response is empty")
            if is_a_change:
                self.has_more_changes = False
            return False

        data = Miner.parse(data)
        if is_a_change:
            if "\"_more_changes\": true" not in data:
                self.has_more_changes = False
                print('No more changes left')

        if data is None:
            print(f"Error: {url} response could not be parsed")
            return False
        if len(data) == 0:
            print(f" {url} returned empty")
            if is_a_change:
                self.has_more_changes = False
            return False

        # print("Dumping response of {0}".format(url))
        self.dump(path=filename, data=data)
        return True

    def change_details_mine(self, sub_directory: str = "change",
                            parameters: Parameters = Parameters(), timeout: int = 60):
        current_dir = os.path.join(self.root, sub_directory)
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

        future_to_url = {}
        current_index = parameters.start_index
        with ThreadPoolExecutor(max_workers=parameters.n_jobs) as executor:
            jobs = parameters.n_jobs
            while self.has_more_changes and (parameters.end_index is None or parameters.end_index == -1 or current_index < parameters.end_index):
                url = self.create_change_details_url(current_index, parameters)
                filename = self.create_change_filename(parameters.status, current_index, parameters.batch_size)

                path = os.path.join(current_dir, filename)
                if not self.replace and os.path.exists(path):
                    print(f"{filename} already exists")
                    current_index += parameters.batch_size
                    continue

                future = executor.submit(self.download, url, timeout, path, True)
                future_to_url[future] = url
                current_index += parameters.batch_size
                jobs -= 1

            # print("Shutting down executor without waiting")
            # executor.shutdown(wait=False)
            # print("Canceling running futures")
            # for future in future_to_url:
            #     future.cancel()

        results = []
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                did_succeed = future.result()
            except Exception as exc:
                print(f"{url} generated an exception: {exc}")
            else:
                results.append((url, did_succeed))
        return results

    def profiles_mine(self, account_ids, sub_directory: str = "profile", timeout: int = 60):
        current_dir = os.path.join(self.root, sub_directory)
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

        with ThreadPoolExecutor(max_workers=8) as executor:
            for account_id in account_ids:
                join_file_path = os.path.join(current_dir, f"profile_{account_id}.json")

                # download join date
                if self.replace or not os.path.exists(join_file_path):
                    print(account_id)
                    url = f"{self.gerrit.value}/accounts/{account_id}/detail"
                    executor.submit(self.download, url, timeout, join_file_path)

    def profile_mine(self, account_id, sub_directory: str = "profile", timeout: int = 60):
        current_dir = os.path.join(self.root, sub_directory)
        if not os.path.isdir(current_dir):
            os.mkdir(current_dir)

        join_file_path = os.path.join(current_dir, f"profile_{account_id}.json")

        # download join date 
        if self.replace or not os.path.exists(join_file_path):
            print(account_id)
            url = f"{self.gerrit.value}/accounts/{account_id}/detail"
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.download, url, timeout, join_file_path)

        # download all changes associated with this account
        # details_file_path = os.path.join(current_dir , f"profile_details_{account_id}.json")
        # if self.replace or not os.path.exists(details_file_path):
        #     url = f"{self.gerrit.value}/changes/?q=is:closed+owner:{account_id}"
        #     with ThreadPoolExecutor(max_workers=1) as executor:
        #         executor.submit(self.download, url, timeout, details_file_path)

    def comment_mine(self, change_number, sub_directory: str = "comment", timeout: int = 60):
        current_directory = os.path.join(self.root, sub_directory)
        if not os.path.exists(current_directory):
            os.mkdir(current_directory)

        file_path = os.path.join(current_directory, f"comment_{change_number}.json")
        if not self.replace and os.path.exists(file_path):
            return

        url = f"{self.gerrit.value}/changes/{change_number}/comments"
        self.download(url, timeout, file_path)

    def comments_mine(self, change_ids, sub_directory: str = "comment", timeout: int = 60):
        current_directory = os.path.join(self.root, sub_directory)
        if not os.path.exists(current_directory):
            os.mkdir(current_directory)

        for change_id in change_ids:
            file_path = os.path.join(current_directory, f"comment_{change_id}.json")
            if not self.replace and os.path.exists(file_path):
                return

            url = f"{self.gerrit.value}/changes/{change_id}/comments"
            self.download(url, timeout, file_path)
