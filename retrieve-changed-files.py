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


class OpenstackChangedLinesRetrieval:
    """
    A class for processing and retrieving OpenStack code changes and associated data.
    Handles API interactions with OpenDev review system to obtain changed files and code diffs.
    """

    def __init__(self, base_dir='.'):
        """
        Initialize the OpenstackChangedLinesRetrieval class.

        Args:
            base_dir (str): Base directory for operations and file storage
        """
        self.base_dir = base_dir
        self.changes_dir = osp.join(base_dir, 'Changes2')
        self.files_dir = osp.join(base_dir, 'Files')
        self.processed_files_path = osp.join(self.files_dir, 'processed_files.csv')
        self.unprocessed_files_path = osp.join(self.files_dir, 'unprocessed_files.csv')

    def retrieve_changed_files(self, project, number, revision):
        """
        Retrieve the list of files changed in a specific revision of a change.

        Args:
            project (str): Project name
            number (str/int): Change number
            revision (str/int): Revision number

        Returns:
            list: List of changed filenames, excluding metadata files
        """
        url = f'https://review.opendev.org/changes/{project}~{number}/revisions/{revision}/files'
        response = rq.get(url)
        response = response.text.split("\n")[1]  # Skip the first line (anti-XSSI prefix)
        response = json.loads(response)
        
        # Filter out special files
        return [f for f in list(response.keys()) if f not in ['/COMMIT_MSG', '/MERGE_LIST']]

    def retrieve_added_lines(self, row):
        """
        Retrieve added and deleted lines for each file in a change.

        Args:
            row (pandas.Series): Row representing a change with project, number, and revisions info

        Returns:
            pandas.Series: Updated row with added and deleted lines, changed files, and files count
        """
        project = urllib.parse.quote(row['project'], safe='')
        nbr = row['number']
        added_lines = ''
        deleted_lines = ''

        # Extract revision from the revisions field (stored as string representation of list)
        revision = ast.literal_eval(row['revisions'])[0]['number']
        
        # Get the list of changed files
        changed_files = self.retrieve_changed_files(project, nbr, revision)
        row['changed_files'] = changed_files
        row['files_count'] = len(changed_files)

        if len(changed_files) == 0:
            row['added_lines'] = None
            row['deleted_lines'] = None
            return row

        # Sort files for consistent processing
        sorted(changed_files)
        
        # Process each file to get added/deleted lines
        for file in changed_files:
            file = urllib.parse.quote(file, safe='')
            url = f'https://review.opendev.org/changes/{project}~{nbr}/revisions/{revision}/files/{file}/diff?intraline&whitespace=IGNORE_NONE'

            change_response = rq.get(url)
            data = change_response.text.split("\n")[1]  # Skip the first line
            data = json.loads(data)
            
            # Extract added and deleted lines from the diff content
            for item in data['content']:
                if 'a' in item.keys():
                    deleted_lines += ' '.join(item['a']) 
                if 'b' in item.keys():
                    added_lines += ' '.join(item['b']) 
        
        # Clean up whitespace in the extracted lines
        row['added_lines'] = re.sub(r'\s+', ' ', added_lines)
        row['deleted_lines'] = re.sub(r'\s+', ' ', deleted_lines)
        return row

    def process_csv_file(self, file_name):
        """
        Process a single CSV file to retrieve added and deleted lines for each change.

        Args:
            file_name (str): Name of the CSV file to process

        Returns:
            str or None: File name if processing was successful, None otherwise
        """
        try:
            print(f'Processing {file_name} file started...')

            file_path = osp.join(self.changes_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Convert string representation of list to actual list
            df['changed_files'] = df['changed_files'].map(ast.literal_eval)
            
            df['added_lines'] = None
            df['deleted_lines'] = None

            # Process each row to retrieve added and deleted lines
            df = df.apply(self.retrieve_added_lines, axis=1)

            # Save the updated dataframe back to the CSV file
            df.to_csv(file_path, index=None)
            
            print(f'File {file_name} processed successfully...')
            
            return file_name
        except Exception as ex:
            print(f'Error while processing the following file: {file_name}')
            print(f'Exception: {ex}')

            # Track unprocessed files
            self.track_unprocessed_file(file_name)
            return None

    def track_unprocessed_file(self, file_name):
        """
        Add a file to the list of unprocessed files.

        Args:
            file_name (str): Name of the file that couldn't be processed
        """
        try:
            unprocessed_files = pd.read_csv(self.unprocessed_files_path)['name'].values.tolist()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            unprocessed_files = []
            
        unprocessed_files.append(file_name)
        pd.DataFrame({'name': unprocessed_files}).to_csv(self.unprocessed_files_path, index=None)

    def copy_unprocessed_files(self, files, source_dir="../openstack-evolution/Changes/"):
        """
        Copy unprocessed files from a source directory to the changes directory.

        Args:
            files (list): List of file names to copy
            source_dir (str): Source directory containing the files
        """
        try:
            for f in files:
                command = ["cp", f"{source_dir}{f}", f"{self.changes_dir}/{f}"]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                print(result)
        except subprocess.CalledProcessError as e:
            print("Error executing the command:", e)

    def process_all_files(self, use_concurrent=True):
        """
        Process all remaining files that haven't been processed yet.

        Args:
            use_concurrent (bool): Whether to use concurrent processing

        Returns:
            list: List of successfully processed files
        """
        # Load list of already processed files
        try:
            df_processed_files = pd.read_csv(self.processed_files_path)
            processed_files = df_processed_files['name'].values.tolist()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df_processed_files = pd.DataFrame(columns=['name'])
            processed_files = []
        
        # Get all files in the changes directory
        all_files = hpr.list_file(self.changes_dir)
        
        # Find files that haven't been processed yet
        remaining_files = [f for f in all_files if f not in processed_files]
        
        if not remaining_files:
            print("No files remaining to process.")
            return processed_files
            
        print(f"Found {len(remaining_files)} files to process.")
        
        if use_concurrent:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(self.process_csv_file, f) for f in remaining_files]

                for f in concurrent.futures.as_completed(results):
                    result = f.result()
                    if result is not None:
                        processed_files.append(result)
                        df_processed_files = pd.DataFrame({'name': processed_files})
                        df_processed_files.to_csv(self.processed_files_path, index=None)
        else:
            for f in remaining_files:
                result = self.process_csv_file(f)
                if result is not None:
                    processed_files.append(result)
                    df_processed_files = pd.DataFrame({'name': processed_files})
                    df_processed_files.to_csv(self.processed_files_path, index=None)
                    
        return processed_files

    def run(self):
        """
        Run the full processing pipeline.
        """
        print(f"Script {__file__} started...")

        start_date, start_header = hpr.generate_date("This script started at")
        
        # Ensure directories exist
        os.makedirs(self.changes_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)
        
        # Process all files
        self.process_all_files()

        end_date, end_header = hpr.generate_date("This script ended at")

        print(start_header)
        print(end_header)
        hpr.diff_dates(start_date, end_date)

        print(f"Script completed\n")


# Example usage
if __name__ == '__main__':
    processor = OpenstackChangedLinesRetrieval()
    processor.run()