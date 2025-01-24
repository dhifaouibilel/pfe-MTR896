import pandas as pd
import numpy as np
import concurrent.futures
import json
import os
import os.path as osp
import shutil
import llmcp.utils.helpers as hpr
import re

DIR = hpr.DIR


def process_json_file(file_name):
    """Transform a json file into a readable python dict format"""
    with open("%sData/%s" % (DIR, file_name), "r") as string:
        dict_data = json.load(string)
    string.close()
    return dict_data


def retrieve_reviewers(df, index):
    """Filter the reviewers of each change"""
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

    filepath = "%sReviewers/reviewers_data_%d.csv" % (DIR, index)
    reviewers_df.to_csv(filepath, index=False, encoding="utf-8")


def retrieve_messages(df, index):
    """Filter the discussion messages of each change"""
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

    filepath = "%sMessages/messages_data_%d.csv" % (DIR, index)
    messages_df.to_csv(filepath, index=False, encoding="utf-8")


def filter_files_attr(revisions):
    """Filter files of the current change"""
    # if row["current_revision"] not in row["revisions"].keys():
    #     return {}

    first_revision = revisions[0]
    return first_revision["files"]


def retrieve_files(df, index):
    """Filter the files of each change"""
    revisions_df = df[["change_id", "current_revision", "project", "subject", "revisions"]].copy()

    revisions_df["files"] = revisions_df.apply(
        lambda row: {
            "change_id": row["change_id"],
            "current_revision": row["current_revision"],
            "project": row["project"],
            "subject": row["subject"],
            "files": filter_files_attr(row),
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

    del files_df["files"]
    del files_df["status"]

    files_df = files_df.drop(columns=["files", "status", "old_path", "binary"], errors="ignore")

    file_path = "%sFilesOS/files_data_%d.csv" % (DIR, index)
    files_df.to_csv(file_path, index=False, encoding="utf-8")


def calc_nbr_files(row):
    """Count number of files for each change"""
    return len(filter_files_attr(row))


def retrieve_commit_message(row):
    """Retrieve commit message of each review"""
    if row["current_revision"] not in row["revisions"].keys():
        keys = list(row["revisions"].keys())
        return row["revisions"][keys[0]]["commit"]["message"]

    return row["revisions"][row["current_revision"]]["commit"]["message"]


def retrieve_git_command(row):
    """Retrieve git command for changed lines of code"""
    revisions = list(row.values())

    fetch = revisions[0]["fetch"]["anonymous http"]

    # git_command = "git fetch {} {} && git format-patch -1 --stdout FETCH_HEAD".format(fetch["url"], fetch["ref"])
    git_command = f'{fetch["commands"]["Pull"]} && git config pull.ff only'
    return git_command


def retrieve_revisions(revisions):
    # revisions = row["revisions"]
    results = []
    # print(revisions.keys())
    for rev in list(revisions.values()):
        # url = None
        # for item in rev['commit']['web_links']:
        #     if item['name'] == 'gitea':
        #         url = item['url']
        #         break

        results.append(
            {
                "number": rev["_number"],
                "created": rev["created"],
                "files": list(rev["files"].keys()) if "files" in rev.keys() else [],
                # "web_link": url,
                "message": rev["commit"]["message"],
            }
        )
    results.sort(key=lambda x: x["created"], reverse=False)

    return results if len(results) > 0 else []


def extract_messages(messages):
    res = [
        {"rev_nbr": msg["_revision_number"], "author": msg["date"], "date": msg["date"]}
        for msg in messages
        if any(item in msg["message"] for item in ["Build failed.", "Build succeeded."])
    ]

    return res if len(res) > 0 else None


# def retrieve_commit_id(revisions):
#     """Retrieve commit id related to a given change
#     """
#     first_revision = revisions[0]
#     return first_revision["commit_id"]


def extract_commit_id(revision):
    """Retrieve commit id related to a given change"""
    url = revision[0]["web_link"]

    pattern = r"/commit/([a-f0-9]+)$"  # Regular expression pattern to match the commit ID

    match = re.search(pattern, url)
    if match:
        commit_id = match.group(1)
        # print("Commit ID:", commit_id)
        return commit_id

    # print("No commit ID found in the URL.")
    return None


def retrieve_changes(file):
    """Filter the changes"""
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
        # "reviewers", "messages",  "total_comment_count",
        "revisions",
        "_number",
        "current_revision",
    ]

    df = pd.read_json("%s/Data2/%s" % (os.getcwd(), file))

    df = df[changes_columns]

    # df["discussion_messages_count"] = df["messages"].copy().map(lambda x: len(x))
    # df["reviewers"] = df["reviewers"].map(lambda x: x["REVIEWER"]
    #   if "REVIEWER" in x.keys() else [])
    # df["reviewers_count"] = df["reviewers"].map(lambda x: len(x))
    # df["revisions_count"] = df["revisions"].map(lambda x: len(x))

    df["owner_account_id"] = df["owner"].map(lambda x: x["_account_id"] if "_account_id" in x.keys() else None)
    df["owner_name"] = df["owner"].map(lambda x: x["name"] if "name" in x.keys() else None)
    df["owner_username"] = df["owner"].map(lambda x: x["username"] if "username" in x.keys() else None)

    df["is_owner_bot"] = df["owner"].map(lambda x: 1 if "tags" in x.keys() else 0)

    df["commit_message"] = df.apply(retrieve_commit_message, axis=1)

    # df["git_command"] = df["revisions"].map(retrieve_git_command)

    df["revisions"] = df["revisions"].map(retrieve_revisions)
    # df["messages"] = df["revisions"].map(extract_messages)
    # df["commit_id"] = df["revisions"].map(retrieve_commit_id)
    # df["changed_files"] = df["revisions"].map(filter_files_attr)
    # df["files_count"] = df["revisions"].map(calc_nbr_files)

    # df['commit_id'] = df["revisions"].map(extract_commit_id)

    # if "topic" in origin_df.columns:
    #     df = pd.concat((df, origin_df[["topic"]]), axis=1)

    del df["owner"]

    changes_df = df.copy()
    changes_df.columns = changes_df.columns.str.replace("_number", "number")

    # del changes_df["reviewers"]
    # del changes_df["messages"]
    # del changes_df["revisions"]

    file_path = osp.join(os.getcwd(), "changes", f"data_{file[5:-5]}.csv")
    changes_df.to_csv(file_path, index=False, encoding="utf-8")

    print(f"{file} file completed successfully...")

    return file


if __name__ == "__main__":
    print("Script openstack-data-transform.py started...")

    start_date, start_header = hpr.generate_date("This script started at")

    changes_dir = "%schanges" % DIR
    # reviewers_dir = "%sReviewers" % DIR
    # messages_dir = "%sMessages" % DIR
    # files_dir = "%sFilesOS" % DIR

    for dir in list(
        [
            changes_dir,  # reviewers_dir, messages_dir, files_dir
        ]
    ):
        if os.path.exists(dir):
            shutil.rmtree(path=dir)
        os.makedirs(dir)

    # index = 0
    # file_path = "openstack_data_722.json"
    json_files = hpr.list_file(osp.join(os.getcwd(), "Data2"))
    processed_files = []
    # processed_files_path = osp.join(os.getcwd(), 'Changes')

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(retrieve_changes, f) for f in json_files]

        for f in concurrent.futures.as_completed(results):
            if f:
                print(f"File {f.result()} processed successfully!")
                processed_files.append(f.result())
                # pd.DataFrame({'name': processed_files}).to_csv(processed_files_path, index=None)

    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print("Script openstack-data-transform.py ended\n")
