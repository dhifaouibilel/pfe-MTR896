import pandas as pd
import os
import os.path as osp
import subprocess
import concurrent.futures
import json
import requests as rq
import ast
import re
import urllib.parse
import utils.helpers as hpr

def retrieve_changed_files(project, number, revision):
    url = f'https://review.opendev.org/changes/{project}~{number}/revisions/{revision}/files'

    response = rq.get(url)

    response = response.text.split("\n")[1]

    response = json.loads(response)

    return [f for f in list(response.keys()) if f not in ['/COMMIT_MSG', '/MERGE_LIST']]


def retrieve_added_lines(row):
    project = urllib.parse.quote(row['project'], safe='')
    nbr = row['number']
    changed_files = row['changed_files']
    added_lines = ''

    revision = ast.literal_eval(row['revisions'])[0]['number']
    changed_files = retrieve_changed_files(project, nbr, revision)
    row['changed_files'] = changed_files
    row['files_count'] = len(changed_files)

    if len(changed_files) == 0:
        row['added_lines'] = None
        row['deleted_lines'] = None
        return row

    sorted(changed_files)
    
    for file in changed_files:
        file = urllib.parse.quote(file, safe='')
        url = f'https://review.opendev.org/changes/{project}~{nbr}/revisions/{revision}/files/{file}/diff?intraline&whitespace=IGNORE_NONE'

        change_response = rq.get(url)

        data = change_response.text.split("\n")[1]

        data = json.loads(data)
        added_lines = ""
        deleted_lines = ""
        for item in data['content']:
            if 'a' in item.keys():
                deleted_lines += ' '.join(item['a']) 
            if 'b' in item.keys():
                added_lines += ' '.join(item['b']) 
    row['added_lines'] = re.sub(r'\s+', ' ', added_lines)
    row['deleted_lines'] = re.sub(r'\s+', ' ', deleted_lines)
    return row

def process_csv_file(file_name):
    
    try:
        print(f'Processing {file_name} file started...')

        file_path = osp.join('.', 'Changes', file_name)
        df = pd.read_csv(file_path)
        # df = df.iloc[:, :-31]
        df['changed_files'] = df['changed_files'].map(ast.literal_eval)
        
        df['added_lines'] = None
        df['deleted_lines'] = None

        df = df.apply(retrieve_added_lines, axis=1)

        df.to_csv(file_path, index=None)
        
        print(f'File {file_name} processed successfully...')
        
        return file_name
    except Exception as ex:
        print(f'Error while processing the following file: {file_name}')

        unprocessed_files_path = osp.join('.', 'Files', 'unprocessed_files.csv')

        unprocessed_files = pd.read_csv(unprocessed_files_path)['name'].values.tolist()

        unprocessed_files.append(file_name)

        pd.DataFrame({'name': unprocessed_files}).to_csv(unprocessed_files_path, index=None)

        return None


def copy_unprocessed_files(files):

    # Use subprocess.run() to execute the command
    try:
        for f in files:
            command = ["cp", f"../openstack-evolution/Changes/{f}", f"./Changes/{f}"]  # Example: list files in the current directory
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            print(result)
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)


if __name__ == '__main__':
    print(f"Script {__file__} started...")

    start_date, start_header = hpr.generate_date("This script started at")
    
    processed_files_path = osp.join('.', 'Files', 'processed_files.csv')
    df_processed_files = pd.read_csv(processed_files_path)
    df_processed_files = df_processed_files[0:0]

    processed_files = df_processed_files['name'].values.tolist()
    
    all_files = hpr.list_file(osp.join(os.getcwd(), 'Changes'))
    
    remaining_files = [f for f in all_files if f not in processed_files]
    # remaining_files = ["changes_data_556.csv"]

    # copy_unprocessed_files(remaining_files)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_csv_file, f) for f in remaining_files]

        for f in concurrent.futures.as_completed(results):
            if f != None:
                processed_files.append(f.result())
                pd.DataFrame({'name': processed_files}).to_csv(processed_files_path, index=None)


    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print(f"Script {__file__} ended\n")