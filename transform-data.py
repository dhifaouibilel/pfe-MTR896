import pandas as pd
import numpy as np
import concurrent.futures
import json
import os
import os.path as osp
import shutil
import utils.helpers as hpr
import re

from logging_config import get_logger

class OpenStackDataTransformer:
    """
    A class for transforming OpenStack data from JSON files to CSV format.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the transformer with a base directory.
        
        Args:
            base_dir: The base directory for data processing. If None, uses the default from helpers.
        """
        self.DIR = base_dir if base_dir is not None else hpr.DIR
        # Folder containing the JSON file (raw data from Gerrit)
        self.DATA_DIR = osp.join('.', 'Data')
        self.CHANGES_DIR = 'Changes2'

        # BILEL (A supprimer apres)
        # Init MongoManager class
        # mogo_manager = MongoManager()

        self.logger = get_logger()
        self.logger.info(f"{self.DATA_DIR=}")
        
    def process_json_file(self, file_name):
        """
        Transform a json file into a readable python dict format.
        
        Args:
            file_name: The name of the JSON file to process.
            
        Returns:
            dict: The loaded JSON data as a Python dictionary.
        """
        with open(f"{self.DIR}Data/{file_name}", "r") as string:
            dict_data = json.load(string)
        string.close()
        return dict_data
    
    def retrieve_reviewers(self, df, index):
        """
        Filter the reviewers of each change.
        
        Args:
            df: DataFrame containing change data.
            index: Index number for the output file.
        """
        old_columns = ["change_id", "reviewers"]
        main_columns = ["change_id", "account_id", "name", "email", "username"]

        reviewers_df = df[old_columns].copy()

        reviewers_df["flatenned_object"] = reviewers_df.apply(
            lambda row: {"change_id": row["change_id"], "reviewers": row["reviewers"]}, axis=1
        )

        reviewers_df = reviewers_df.drop(columns=old_columns)

        reviewers_df = pd.json_normalize(
            data=reviewers_df["flatenned_object"], record_path="reviewers", meta=["change_id"], sep="_", errors="ignore"
        )

        reviewers_df.columns = reviewers_df.columns.str.replace("_account_id", "account_id")

        reviewers_df = reviewers_df[main_columns]

        filepath = f"{self.DIR}Reviewers/reviewers_data_{index}.csv"
        reviewers_df.to_csv(filepath, index=False, encoding="utf-8")
    
    def retrieve_messages(self, df, index):
        """
        Filter the discussion messages of each change.
        
        Args:
            df: DataFrame containing change data.
            index: Index number for the output file.
        """
        old_columns = ["change_id", "current_revision", "messages"]

        new_columns = [
            "change_id",
            "id",
            "date",
            "message",
            "author_account_id",
            "author_name",
            "author_username",
            "real_author_account_id",
            "real_author_name",
            "real_author_username",
            "author_email",
            "real_author_email",
        ]

        messages_df = df[old_columns].copy()

        messages_df["flatenned_object"] = messages_df.apply(
            lambda row: {"change_id": row["change_id"], "messages": row["messages"]}, axis=1
        )

        for c in old_columns:
            del messages_df[c]

        messages_df = pd.json_normalize(
            data=messages_df["flatenned_object"], record_path="messages", meta=["change_id"], sep="_", errors="ignore"
        )

        messages_df = messages_df.rename(
            columns={"author__account_id": "author_account_id", "real_author__account_id": "real_author_account_id"}
        )

        messages_df = messages_df[new_columns]

        filepath = f"{self.DIR}Messages/messages_data_{index}.csv"
        messages_df.to_csv(filepath, index=False, encoding="utf-8")
    
    def filter_files_attr(self, revisions):
        """
        Filter files of the current change.
        
        Args:
            revisions: List of revisions.
            
        Returns:
            dict: Dictionary of files.
        """
        first_revision = revisions[0]
        return first_revision["files"]
    
    def retrieve_files(self, df, index):
        """
        Filter the files of each change.
        
        Args:
            df: DataFrame containing change data.
            index: Index number for the output file.
        """
        revisions_df = df[["change_id", "current_revision", "project", "subject", "revisions"]].copy()

        revisions_df["files"] = revisions_df.apply(
            lambda row: {
                "change_id": row["change_id"],
                "current_revision": row["current_revision"],
                "project": row["project"],
                "subject": row["subject"],
                "files": self.filter_files_attr(row["revisions"]),
            },
            axis=1,
        )

        files_df = revisions_df[["files"]]

        files_data = []

        for row in np.array(files_df):
            row = row[0]
            file_keys = row["files"].keys()

            if len(file_keys) == 0:
                continue

            for fk in file_keys:
                new_row = {"name": fk, **row, **row["files"][fk]}
                files_data.append(new_row)

        files_df = pd.DataFrame(files_data)

        # Safely remove columns that might not exist
        files_df = files_df.drop(columns=["files", "status", "old_path", "binary"], errors="ignore")

        file_path = f"{self.DIR}FilesOS/files_data_{index}.csv"
        files_df.to_csv(file_path, index=False, encoding="utf-8")

    def calc_nbr_files(self, row):
        """
        Count number of files for each change.
        
        Args:
            row: Row from DataFrame.
            
        Returns:
            int: Number of files.
        """
        return len(self.filter_files_attr(row["revisions"]))
    
    def retrieve_commit_message(self, row):
        """
        Retrieve commit message of each review.
        
        Args:
            row: Row from DataFrame.
            
        Returns:
            str: Commit message.
        """
        if row["current_revision"] not in row["revisions"].keys():
            keys = list(row["revisions"].keys())
            return row["revisions"][keys[0]]["commit"]["message"]

        return row["revisions"][row["current_revision"]]["commit"]["message"]
    
    def retrieve_git_command(self, row):
        """
        Retrieve git command for changed lines of code.
        
        Args:
            row: Row from DataFrame.
            
        Returns:
            str: Git command.
        """
        revisions = list(row.values())
        fetch = revisions[0]["fetch"]["anonymous http"]
        git_command = f'{fetch["commands"]["Pull"]} && git config pull.ff only'
        return git_command
    
    def retrieve_revisions(self, revisions):
        """
        Process revisions data.
        
        Args:
            revisions: Revisions data.
            
        Returns:
            list: List of processed revisions.
        """
        results = []
        for rev in list(revisions.values()):
            results.append(
                {
                    "number": rev["_number"],
                    "created": rev["created"],
                    "files": list(rev["files"].keys()) if "files" in rev.keys() else [],
                    "message": rev["commit"]["message"],
                }
            )
        results.sort(key=lambda x: x["created"], reverse=False)

        return results if len(results) > 0 else []
    
    def extract_messages(self, messages):
        """
        Extract specific messages from change data.
        
        Args:
            messages: List of messages.
            
        Returns:
            list or None: Filtered messages or None if empty.
        """
        res = [
            {"rev_nbr": msg["_revision_number"], "author": msg["date"], "date": msg["date"]}
            for msg in messages
            if any(item in msg["message"] for item in ["Build failed.", "Build succeeded."])
        ]

        return res if len(res) > 0 else None
    
    def extract_commit_id(self, revision):
        """
        Retrieve commit id related to a given change.
        
        Args:
            revision: Revision data.
            
        Returns:
            str or None: Commit ID or None if not found.
        """
        url = revision[0]["web_link"]
        pattern = r"/commit/([a-f0-9]+)$"  # Regular expression pattern to match the commit ID
        match = re.search(pattern, url)
        
        if match:
            commit_id = match.group(1)
            return commit_id
        
        return None
    
    def retrieve_changes(self, file):
        """
        Filter the changes from a JSON file and save to CSV.
        
        Args:
            file: The name of the JSON file to process.
            
        Returns:
            str: The processed file name.
        """
        print(f"Processing {file} file started...")

        changes_columns = [
            "id",
            "project",
            "branch",
            "change_id",
            "owner",
            "subject",
            "status",
            "created",
            "updated",  # "submitted",
            "added_lines",
            "deleted_lines",
            "reviewers", "messages", "total_comment_count",
            "revisions",
            "_number",
            "current_revision",
        ]

        # Read JSON data
        df = pd.read_json(f"{self.DATA_DIR}/{file}")
        df = df[changes_columns]

        # Process reviewers
        df["reviewers"] = df["reviewers"].map(lambda x: x["REVIEWER"] if "REVIEWER" in x.keys() else [])
        
        # Process owner information
        df["owner_account_id"] = df["owner"].map(lambda x: x["_account_id"] if "_account_id" in x.keys() else None)
        df["owner_name"] = df["owner"].map(lambda x: x["name"] if "name" in x.keys() else None)
        df["owner_username"] = df["owner"].map(lambda x: x["username"] if "username" in x.keys() else None)
        df["is_owner_bot"] = df["owner"].map(lambda x: 1 if "tags" in x.keys() else 0)
        
        # Process commit and revision information
        df["commit_message"] = df.apply(self.retrieve_commit_message, axis=1)
        df["revisions"] = df["revisions"].map(self.retrieve_revisions)
        df["messages"] = df["messages"].map(self.extract_messages)
        df["changed_files"] = df["revisions"].map(self.filter_files_attr)
        
        # Remove owner column as it's been processed
        del df["owner"]

        df_changes = df.copy()
        df_changes.columns = df_changes.columns.str.replace("_number", "number")

        # Save to CSV
        # file_path = osp.join('.', self.CHANGES_DIR, f"data_{file.split('_')[-1].split('.')[0]}.csv")
        # df_changes.to_csv(file_path, index=False, encoding="utf-8")

        # BILEL (A supprimer apres)
        # Invoke a method to save dataframe to Mongo DB
        # mongo_manager.save_changes_to_dv(df_changes)

        print(f"{file} file completed successfully...")

        return file
    
    def transform_all_data(self):
        """
        Main method to transform all JSON files to CSV.
        """
        print("OpenStack Data Transformation started...")

        start_date, start_header = hpr.generate_date("This transformation started at")

        # Create or clean directories
        changes_dir = f"{self.DIR}/{self.CHANGES_DIR}"
        # reviewers_dir = f"{self.DIR}Reviewers"
        # messages_dir = f"{self.DIR}Messages"
        # files_dir = f"{self.DIR}FilesOS"

        for dir_path in [changes_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(path=dir_path)
            os.makedirs(dir_path)

        # Get list of JSON files
        json_files = hpr.list_file(self.DATA_DIR)
        processed_files = []

        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self.retrieve_changes, f) for f in json_files]

            for f in concurrent.futures.as_completed(results):
                if f:
                    print(f"File {f.result()} processed successfully!")
                    processed_files.append(f.result())

        end_date, end_header = hpr.generate_date("This transformation ended at")

        print(start_header)
        print(end_header)
        hpr.diff_dates(start_date, end_date)

        print("OpenStack Data Transformation completed\n")
        
        return processed_files


# if __name__ == "__main__":
#     transformer = OpenStackDataTransformer()
#     transformer.transform_all_data()