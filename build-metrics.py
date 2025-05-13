import os
import os.path as osp
from datetime import timedelta
import pandas as pd
import numpy as np
import concurrent.futures
import ast
import re
from typing import List, Dict, Set, Optional, Union
import utils.helpers as hpr
from utils import constants

class ChangeMetricsCalculator:
    """
    A class to calculate various metrics for code changes in software projects.
    """
    
    DESCRIPTION_METRICS = constants.DESCRIPTION_METRICS
    
    def __init__(self):
        """Initialize the calculator with empty data structures."""
        self.df_changes: Optional[pd.DataFrame] = None
        self.df_deps: Optional[pd.DataFrame] = None
        self.dependent_changes: Optional[Set[int]] = None
        self.cross_pro_changes: Optional[Set[int]] = None
        self.within_pro_changes: Optional[Set[int]] = None
    
    def init_global_vars(self) -> None:
        """
        Initialize the global variables by loading and preprocessing the data.
        """
        df = hpr.combine_openstack_data(changes_path="/Changes3/")
        df = df[['number', 'project', 'created', 'owner_account_id', 'branch', 
                'changed_files', 'insertions', 'deletions', 'subject', 
                'commit_message', 'status']]
        df['changed_files'] = df['changed_files'].map(ast.literal_eval)
        
        df_dep_changes = pd.read_csv(osp.join('.', 'Files', 'all_dependencies.csv'))
        df_dep_changes = df_dep_changes[(df_dep_changes['Source_status']!='NEW') & 
                                       (df_dep_changes['Target_status']!='NEW')]
        df_dep_changes['Source_date'] = pd.to_datetime(df_dep_changes['Source_date'])
        df_dep_changes['Target_date'] = pd.to_datetime(df_dep_changes['Target_date'])
        
        dependent_changes_loc = set(hpr.flatten_list(df_dep_changes[['Source', 'Target']].values))
        cross_pro_changes_loc = set(hpr.flatten_list(
            df_dep_changes.loc[df_dep_changes['is_cross']=='Cross', ['Source', 'Target']].values))
        within_pro_changes_loc = dependent_changes_loc.difference(cross_pro_changes_loc)
        
        df['is_dependent'] = df['number'].map(lambda nbr: nbr in dependent_changes_loc)
        df['is_cross'] = df['number'].map(lambda nbr: nbr in cross_pro_changes_loc)
        
        self.df_changes = df
        self.df_deps = df_dep_changes
        self.dependent_changes = dependent_changes_loc
        self.cross_pro_changes = cross_pro_changes_loc
        self.within_pro_changes = within_pro_changes_loc
    
    # Project-related metrics
    def count_project_age(self, row: pd.Series) -> float:
        """Calculate the age of the project when the change was created."""
        project_creation_date = self.df_changes.loc[
            self.df_changes['project'] == row['project'], 'created'].iloc[0]
        return (row['created'] - project_creation_date).total_seconds()
    
    def count_project_changes(self, row: pd.Series) -> int:
        """Count number of changes in the project before this change."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['created'] < row['created']), 
            'number'
        ].nunique()
    
    def count_whole_changes(self, row: pd.Series) -> int:
        """Count total number of changes across all projects before this change."""
        return self.df_changes.loc[
            self.df_changes['created'] < row['created'], 
            'number'
        ].nunique()
    
    # Developer-related metrics
    def count_project_changes_owner(self, row: pd.Series) -> int:
        """Count changes by the same owner in the same project before this change."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_whole_changes_owner(self, row: pd.Series) -> int:
        """Count all changes by the same owner before this change."""
        return self.df_changes.loc[
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_projects_contributed(self, row: pd.Series) -> int:
        """Count number of projects the owner has contributed to before this change."""
        return self.df_changes.loc[
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'project'
        ].nunique()
    
    # Dependency-related metrics
    def count_projects_changes_deps(self, row: pd.Series) -> int:
        """Count dependent changes in the same project by the same owner."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_cross_project_changes(self, row: pd.Series) -> int:
        """Count cross-project changes in the same project before this change."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['is_cross'] == True) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_within_project_changes(self, row: pd.Series) -> int:
        """Count within-project dependent changes before this change."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['is_cross'] == False) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_whole_within_project_changes(self, row: pd.Series) -> int:
        """Count all within-project dependent changes across projects before this change."""
        return self.df_changes.loc[
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['is_cross'] == False) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_cross_pro_changes_owner(self, row: pd.Series) -> int:
        """Count cross-project changes by the same owner in the same project."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['is_cross'] == True) &
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_within_pro_changes_owner(self, row: pd.Series) -> int:
        """Count within-project changes by the same owner in the same project."""
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['is_cross'] == False) &
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_last_x_days_dependent_projects(self, row: pd.Series, days: int) -> int:
        """Count dependent projects in the last X days."""
        days_ago = row['created'] - timedelta(days=days)
        projects = set(hpr.flatten_list(self.df_deps.loc[
            (
                ((self.df_deps['Source_repo'] == row['project']) &
                (self.df_deps['Source_date'] >= days_ago) &
                (self.df_deps['Source_date'] < row['created'])) |
                (self.df_deps['Target_repo'] == row['project']) &
                (self.df_deps['Target_date'] >= days_ago) &
                (self.df_deps['Target_date'] < row['created'])
            ) &
            (self.df_deps['is_cross'] == 'Cross'),
            ['Target_repo', 'Source_repo']
        ].values))
        return len(projects) - 1 if projects else 0
    
    def count_avg_cro_proj_nbr(self, row: pd.Series) -> float:
        """Calculate average number of cross-project changes per project."""
        df_deps_sub = self.df_deps.loc[
            (self.df_deps['is_cross'] == 'Cross') &
            (
                ((self.df_deps['Source_repo'] == row['project']) & 
                (self.df_deps['Source_date'] < row['created'])
            ) |
            ((self.df_deps['Target_repo'] == row['project']) & 
             (self.df_deps['Target_date'] < row['created']))
            ),
            ['Target_repo', 'Source_repo', 'is_cross']
        ]
        df_deps_sub['project'] = df_deps_sub.apply(
            lambda x: x['Target_repo'] if row['project'] == x['Source_repo'] else x['Source_repo'], 
            axis=1)
        
        cross_project_changes_per_project = df_deps_sub.groupby('project')['is_cross'].sum()
        return cross_project_changes_per_project.mean()
    
    def count_last_x_days_cross_project_changes(self, row: pd.Series, days: int) -> int:
        """Count cross-project changes in the last X days."""
        days_ago = row['created'] - timedelta(days=days)
        return self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['is_cross'] == True) &
            (self.df_changes['created'] >= days_ago) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    def count_last_x_days_modified_unique_projects(self, row: pd.Series, days: int) -> int:
        """Count unique projects modified in the last X days."""
        days_ago = row['created'] - timedelta(days=days)
        return self.df_changes.loc[
            (self.df_changes['created'] >= days_ago) &
            (self.df_changes['created'] < row['created']),
            'project'
        ].nunique()
    
    def count_whole_changes_deps(self, row: pd.Series) -> int:
        """Count all dependent changes by the same owner before this change."""
        return self.df_changes.loc[
            (self.df_changes['number'].isin(self.dependent_changes)) &
            (self.df_changes['owner_account_id'] == row['owner_account_id']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    # File-related metrics
    @staticmethod
    def count_num_file_types(changed_files: List[str]) -> int:
        """Count number of unique file extensions in changed files."""
        return len({os.path.splitext(f)[1].lower() for f in changed_files})
    
    @staticmethod
    def count_num_directory_files(changed_files: List[str]) -> int:
        """Count number of unique directories in changed files."""
        return len({os.path.splitext(f)[0].lower() for f in changed_files})
    
    # Text-related metrics
    @staticmethod
    def count_desc_length(desc: str) -> int:
        """Count length of description."""
        return len(desc)
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count number of words in text."""
        return len(re.sub(r'\s+', ' ', text).split())
    
    def identify_desc_nature(self, desc: str, keyword: str) -> int:
        """Identify if description contains specific keywords."""
        words = re.sub(r'\s+', ' ', desc.lower()).split()
        change_type = self.DESCRIPTION_METRICS[keyword]
        return int(any(
            word for word in words
            if word in change_type['inclusion'] or
            (change_type['pattern'] and re.search(change_type['pattern'], word))
        ))
    
    # Developer-file interaction metrics
    def count_num_dev_modified_files(self, row: pd.Series) -> int:
        """Count number of developers who modified the same files."""
        res = self.df_changes.loc[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['created'] <= row['created']), 
            ["changed_files", "owner_account_id"]
        ]
        res = res.explode('changed_files')
        res = res[res['changed_files'].isin(row['changed_files'])]
        return res['owner_account_id'].nunique()
    
    def count_avg_num_dev_modified_files(self, row: pd.Series) -> float:
        """Calculate average number of developers per modified file."""
        res = self.df_changes[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['created'] <= row['created'])
        ]
        res = res.explode('changed_files')
        res = res[res['changed_files'].isin(row['changed_files'])]
        num_devs = res['owner_account_id'].nunique()
        return num_devs / len(row['changed_files']) if row['changed_files'] else 0
    
    def count_ratio_dep_chan_owner(self, row: pd.Series) -> float:
        """Calculate ratio of dependent changes by owner in project vs all projects."""
        owner_pro_cha = self.df_changes.loc[
           (self.df_changes['project'] == row['project']) &
           (self.df_changes['owner_account_id'] == row['owner_account_id']) &
           (self.df_changes['is_dependent'] == True) &
           (self.df_changes['created'] <= row['created']),
           "number"
        ].nunique()
        
        owner_all_cha = self.df_changes.loc[
           (self.df_changes['owner_account_id'] == row['owner_account_id']) &
           (self.df_changes['is_dependent'] == True) &
           (self.df_changes['created'] <= row['created']),
           "number"
        ].nunique()
        
        return 100 * (owner_pro_cha / owner_all_cha) if owner_all_cha else 0
    
    def count_num_modified_file_dependent_changes(self, row: pd.Series, attr: str) -> str:
        """Count dependent changes that modified the same files."""
        changed_files = row['changed_files']
        sub_operations = {"min": min, "max": max, "median": np.median, "mean": np.mean}
        
        if not changed_files:
            for label in sub_operations:
                row[f"{label}{attr}"] = 0
            row[attr] = 0
            return row['number']
        
        file_counts = {f: 0 for f in changed_files}
        for f in file_counts:
            file_counts[f] = self.df_changes.loc[
                (self.df_changes['is_dependent'] == True) &
                (self.df_changes['project'] == row['project']) &
                (self.df_changes['created'] < row['created']) &
                (self.df_changes['changed_files'].apply(lambda x: f in x)), 
                "number"
            ].nunique()
        
        for label, func in sub_operations.items():
            row[f"{label}{attr}"] = func(list(file_counts.values()))
        
        row[attr] = len([count for count in file_counts.values() if count > 0])
        pd.DataFrame({k: [v] for k, v in row.items()}).to_csv(
            osp.join('.', 'Files', 'Metrics', attr, f"{row['number']}.csv"))
        return row['number']
    
    def count_num_file_changes(self, row: pd.Series) -> int:
        """Count number of changes that modified the same files."""
        res = self.df_changes[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['created'] <= row['created'])
        ]
        res = res.explode('changed_files')
        return res.loc[res['changed_files'].isin(row['changed_files']), 'number'].nunique()
    
    @staticmethod
    def count_num_build(row: pd.Series, attr: str) -> int:
        """Count build-related messages."""
        messages = row['messages']
        if not isinstance(messages, list):
            return 0
        return sum(1 for msg in messages if attr in msg['message'])
    
    def count_num_recent_branch_files_changes(self, row: pd.Series) -> int:
        """Count recent branch file changes."""
        res = self.df_changes[
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['created'] <= row['created'])
        ]
        res = res.explode('changed_files')
        return res.loc[res['changed_files'].isin(row['changed_files']), 'branch'].nunique()
    
    def count_type_changes(self, row: pd.Series, change_type: str) -> int:
        """Count changes of specific type before this change."""
        return self.df_changes.loc[
            (self.df_changes['status'] == change_type) &
            (self.df_changes['project'] == row['project']) &
            (self.df_changes['created'] < row['created']),
            'number'
        ].nunique()
    
    @staticmethod
    def get_path(metric: str) -> str:
        """Get file path for a metric."""
        return osp.join('.', 'Files', f'{metric}.csv')
    
    def process_metrics(self, attr: str) -> str:
        """Process and calculate a specific metric."""
        print(f'Processing {attr} metric started...')
        main_dev_attr = ['project', 'created', 'changed_files', 'owner_account_id', 'status']
        
        metric_processors = {
            # Project metrics
            'project_age': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_project_age, axis=1)}),
            'project_changes_count': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_project_changes, axis=1)}),
            'whole_changes_count': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_whole_changes, axis=1)}),
            'cross_project_changes': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_cross_project_changes, axis=1)}),
            'within_project_changes': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_within_project_changes, axis=1)}),
            'whole_within_project_changes': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_whole_within_project_changes, axis=1)}),
            'last_mth_dep_proj_nbr': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_last_x_days_dependent_projects, args=(30,), axis=1)}),
            'avg_cro_proj_nbr': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_avg_cro_proj_nbr, axis=1)}),
            'last_mth_cro_proj_nbr': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_last_x_days_cross_project_changes, args=(30,), axis=1)}),
            'last_mth_mod_uniq_proj_nbr': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_last_x_days_modified_unique_projects, args=(30,), axis=1)}),
            
            # Developer metrics
            'project_changes_owner': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_project_changes_owner, axis=1)}),
            'whole_changes_owner': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_whole_changes_owner, axis=1)}),
            'projects_contributed': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_projects_contributed, axis=1)}),
            'projects_changes_deps': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_projects_changes_deps, axis=1)}),
            'whole_changes_deps': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_whole_changes_deps, axis=1)}),
            'cross_project_changes_owner': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_cross_pro_changes_owner, axis=1)}),
            'within_project_changes_owner': lambda: self.df_changes.assign(**{attr: self.df_changes.apply(self.count_within_pro_changes_owner, axis=1)}),
            'ratio_dep_chan_owner': lambda: self.df_changes.assign(**{attr: self.df_changes[main_dev_attr + ['number']].apply(self.count_ratio_dep_chan_owner, axis=1)}),
            
            # Change metrics
            'code_churn': lambda: self.df_changes.assign(code_churn=self.df_changes['insertions'] + self.df_changes['deletions']),
            
            # File metrics
            'num_file_types': lambda: self.df_changes.assign(num_file_types=self.df_changes['changed_files'].map(self.count_num_file_types)),
            'num_directory_files': lambda: self.df_changes.assign(num_directory_files=self.df_changes['changed_files'].map(self.count_num_directory_files)),
            'num_dev_modified_files': lambda: self.df_changes.assign(**{attr: self.df_changes[main_dev_attr].apply(self.count_num_dev_modified_files, axis=1)}),
            'avg_num_dev_modified_files': lambda: self.df_changes.assign(**{attr: self.df_changes[main_dev_attr].apply(self.count_avg_num_dev_modified_files, axis=1)}),
            'num_file_changes': lambda: self.df_changes.assign(**{attr: self.df_changes[main_dev_attr + ['number']].apply(self.count_num_file_changes, axis=1)}),
            'num_merged_changes': lambda: self.df_changes.assign(**{attr: self.df_changes[main_dev_attr + ['number']].apply(self.count_type_changes, args=('MERGED',), axis=1)}),
            'num_abandoned_changes': lambda: self.df_changes.assign(**{attr: self.df_changes[main_dev_attr + ['number']].apply(self.count_type_changes, args=('ABANDONED',), axis=1)}),
            'num_mod_file_dep_cha': lambda: self.process_complex_metric(main_dev_attr, attr),
            
            # Text metrics
            'subject_length': lambda: self.df_changes.assign(**{attr: self.df_changes['subject'].map(len)}),
            'description_length': lambda: self.df_changes.assign(**{attr: self.df_changes['commit_message'].map(self.count_desc_length)}),
            'subject_word_count': lambda: self.df_changes.assign(**{attr: self.df_changes['subject'].map(self.count_words)}),
            'description_word_count': lambda: self.df_changes.assign(**{attr: self.df_changes['commit_message'].map(self.count_words)}),
        }
        
        if attr in self.DESCRIPTION_METRICS:
            self.df_changes[attr] = self.df_changes['commit_message'].apply(
                lambda msg: self.identify_desc_nature(msg, attr))
        elif attr in metric_processors:
            metric_processors[attr]()
        else:
            raise ValueError(f"Unknown metric: {attr}")
        
        self.save_metric(attr, main_dev_attr)
        print(f'{attr}.csv file saved successfully...')
        return attr
    
    def process_complex_metric(self, main_dev_attr: List[str], attr: str) -> None:
        """Process complex metrics that require parallel processing."""
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [
                executor.submit(self.count_num_modified_file_dependent_changes, row, attr) 
                for _, row in self.df_changes[main_dev_attr + ['number']].iterrows()
            ]
            for future in concurrent.futures.as_completed(results):
                future.result()
                print(f'{future.result()} ID processed successfully...')
    
    def save_metric(self, attr: str, main_dev_attr: List[str]) -> None:
        """Save the calculated metric to a CSV file."""
        compact_columns = ['num_mod_file_dep_cha']
        if attr in compact_columns:
            columns = [f"{subcol}{attr}" for subcol in ['min', 'max', 'mean', 'median']] + ['number', attr]
        else:
            columns = ["number", attr]
        
        self.df_changes[columns].to_csv(
            osp.join('.', 'Files', 'Metrics', f'{attr}.csv'), index=None)

def main():
    """Main function to execute the metrics calculation."""
    print(f"Script {__file__} started...")
    start_date, start_header = hpr.generate_date("This script started at")
    
    calculator = ChangeMetricsCalculator()
    calculator.init_global_vars()
    
    metrics = (
        constants.CHANGE_METRICS + constants.TEXT_METRICS + 
        constants.PROJECT_METRICS + constants.FILE_METRICS + 
        constants.DEVELOPER_METRICS
    )
    
    # Process first metric sequentially
    calculator.process_metrics(metrics[0])
    
    # Process remaining metrics in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(calculator.process_metrics, m) for m in metrics[1:]]
        for future in concurrent.futures.as_completed(results):
            attr = future.result()
            print(f'{attr} metric processed successfully...')
    
    end_date, end_header = hpr.generate_date("This script ended at")
    print(start_header)
    print(end_header)
    hpr.diff_dates(start_date, end_date)
    print(f"Script {__file__} ended\n")

if __name__ == '__main__':
    main()