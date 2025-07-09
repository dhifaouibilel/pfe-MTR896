import os.path as osp
import pandas as pd
from datetime import timedelta
import concurrent.futures
import ast
from utils import helpers as hpr
from utils import constants
from utils import classifier_util as clas_util
from logging_config import get_logger
from db.db_manager import MongoManager


logger = get_logger()

class PairMetricsGenerator:
    def __init__(self):
        self.METRICS = [m for m in constants.get_metrics()[:-8]]
        self.df_dependent_changes = None
        self.dependent_changes = None
        self.cross_pro_changes = None
        self.within_pro_changes = None
        self.df = None
        self.df_changes = None
        self.changed_files = None
        self.changes_description = None
        self.added_lines = None
        self.deleted_lines = None
        self.changes_collection = 'changes'
        self.metrics_collection = 'metrics'
        self.deps_collection = 'dependencies2'
        self.mongo_manager = MongoManager()

    def calc_mod_file_dep_cha(self, row):
        changed_files = row["changed_files"]
        if type(changed_files) is not list:
            changed_files = []
        return round(100*row['num_mod_file_dep_cha']/len(changed_files), 2) if len(changed_files) != 0 else 0
    
    def initialize_global_vars(self):
        """Initialize all global variables needed for analysis"""
        # Load dependencies data
        # df_deps = pd.read_csv(osp.join('.', 'Files', 'source_target_evolution_clean.csv'))
        df_deps = self.mongo_manager.read_all(self.deps_collection)
        df_deps['Source_date'] = pd.to_datetime(df_deps['Source_date'])
        df_deps['Target_date'] = pd.to_datetime(df_deps['Target_date'])
        df_deps['related'] = True
        df_deps.loc[df_deps['Source_repo'].str.startswith('openstack/'), 'Source_repo'] = df_deps['Source_repo'].map(lambda x: x[10:])
        df_deps.loc[df_deps['Target_repo'].str.startswith('openstack/'), 'Target_repo'] = df_deps['Target_repo'].map(lambda x: x[10:])

        self.df_dependent_changes = df_deps

        # Extract sets of different types of changes
        dependent_changes_loc = set(hpr.flatten_list(df_deps[['Source', 'Target']].values))
        cross_pro_changes_loc = set(hpr.flatten_list(df_deps.loc[df_deps['is_cross']==True, ['Source', 'Target']].values))
        within_pro_changes_loc = dependent_changes_loc.difference(cross_pro_changes_loc)

        self.dependent_changes = dependent_changes_loc
        self.cross_pro_changes = cross_pro_changes_loc
        self.within_pro_changes = within_pro_changes_loc

        # Process OpenStack data

        # df_changes = hpr.combine_openstack_data(changes_path="/Changes3/")
        df_changes = self.mongo_manager.read_filtered_changes()

        def safe_literal_eval(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return None
            return x  # déjà un objet Python

        df_changes['reviewers'] = df_changes['reviewers'].map(safe_literal_eval)
        # df_changes['reviewers'] = df_changes['reviewers'].map(ast.literal_eval)
        df_changes['reviewers'] = df_changes['reviewers'].map( lambda x: [rev['_account_id'] for rev in x] if isinstance(x, list) else [])

        # df_changes['reviewers'] = df_changes['reviewers'].map(lambda x: [rev['_account_id'] for rev in x])
        df_changes['changed_files'] = df_changes['changed_files'].map(hpr.combine_changed_file_names)
        df_changes['commit_message'] = df_changes['commit_message'].map(hpr.preprocess_change_description)

        # Combine features
        # df_features = self.combine_features()
        df_features = self.mongo_manager.read_all(self.metrics_collection)
        df_features = pd.merge(
            left=df_features, 
            right=df_changes[['number', 'created', 'project', 'owner_account_id', 'reviewers']], 
            left_on='number', 
            right_on='number', 
            how='inner',
            suffixes=('_source', '_target')
        )

        # Combine file metrics
        # path = osp.join(".", "Files", "file-metrics")
        # file_metrics = hpr.combine_file_metrics(path)

        # # Combine file metrics with original list of features
        # df_features = pd.merge(
        #     left=df_features,
        #     right=file_metrics,
        #     left_on='number',
        #     right_on='number',
        #     how='left',
        #     suffixes=('_source', '_target')
        # )
        # df_features['pctg_mod_file_dep_cha'] = df_features.apply(self.calc_mod_file_dep_cha, axis=1)

        df_features['is_dependent'] = df_features['number'].map(lambda nbr: 1 if nbr in dependent_changes_loc else 0)
        df_features['is_cross'] = df_features['number'].map(self.is_cross_project)

        # Create dictionaries for quick lookups
        self.changed_files = dict(zip(df_changes['number'], df_changes['changed_files']))
        self.changes_description = dict(zip(df_changes['number'], df_changes['commit_message']))
        self.added_lines = dict(zip(df_changes['number'], df_changes['insertions']))
        self.deleted_lines = dict(zip(df_changes['number'], df_changes['deletions']))
        
        self.df_changes = df_changes

        self.df = df_features
    
    def combine_features(self):
        """Combine metrics from multiple files into a single dataframe"""
        metric_path = osp.join('.', 'Files', 'Metrics')
        metric_list = hpr.list_file(metric_path)
        metric_list = [m for m in metric_list if m not in ['num_mod_file_dep_cha', 'num_build_failures.csv']]
        
        df = pd.read_csv(f'{metric_path}/{metric_list[0]}')
        for metric_file in metric_list[1:]:
            df_metric = pd.read_csv(f'{metric_path}/{metric_file}') 
            # Join source and target changes with features of changes
            df = pd.merge(
                left=df, 
                right=df_metric, 
                left_on='number', 
                right_on='number', 
                how='inner',
                suffixes=('_target', '_source')
            )

        df['project_age'] /= (60 * 60 * 24)  # Convert to days

        # Calculate percentages
        df['pctg_cross_project_changes'] = df.apply(self.compute_pctg_cross_project_changes, axis=1)
        # df['pctg_whole_cross_project_changes'] = df.apply(self.compute_pctg_whole_cross_project_changes, axis=1)
        df['pctg_cross_project_changes_owner'] = df.apply(self.compute_ptg_cross_project_changes_owner, axis=1)
        
        return df
    
    def compute_pctg_cross_project_changes(self, row):
        """Calculate percentage of cross-project changes"""
        dominator = row['cross_project_changes'] + row['within_project_changes']
        if dominator == 0:
            return 0
        return row['cross_project_changes'] / dominator

    def compute_pctg_whole_cross_project_changes(self, row):
        """Calculate percentage of whole cross-project changes"""
        dominator = row['whole_cross_project_changes'] + row['whole_within_project_changes']
        if dominator == 0:
            return 0
        return row['whole_cross_project_changes'] / dominator

    def compute_ptg_cross_project_changes_owner(self, row):
        """Calculate percentage of cross-project changes by owner"""
        dominator = row['cross_project_changes_owner'] + row['within_project_changes_owner']
        if dominator == 0:
            return 0
        return row['cross_project_changes_owner'] / dominator
    
    def is_cross_project(self, number):
        """Determine if a change is cross-project, within-project, or neither"""
        if number in self.cross_pro_changes:
            return 1
        elif number in self.within_pro_changes:
            return 0
        else:
            return 2
    
    def compute_common_dev_pctg(self, row):
        """Calculate percentage of common developers between source and target projects"""
        dev_source = self.df.loc[
            (self.df['project'] == row['project_source']) &
            (self.df['created'] < row['Target_date']),
            'owner_account_id'
        ].unique()
        dev_target = self.df.loc[
            (self.df['project'] == row['project_target']) &
            (self.df['created'] < row['Target_date']),
            'owner_account_id'
        ].unique()

        union = len(set(dev_source).union(dev_target))
        intersect = len(set(dev_source).intersection(dev_target))
        return intersect/union if union != 0 else 0

    def count_dev_in_src_change(self, row):
        """Count developer's changes in the source project"""
        changes_nbr = self.df.loc[
            (self.df['project'] == row['project_source']) &
            (self.df['created'] < row['Target_date']) &
            (self.df['owner_account_id'] == row['Target_author']),
            'number'
        ].nunique()

        return changes_nbr

    def count_rev_in_src_change(self, row):
        """Count reviewer's changes in the source project"""
        account_id = row['Target_author']
        reviewers = self.df.loc[
            (self.df['project'] == row['project_source']) &
            (self.df['created'] < row['Target_date']) & 
            (self.df['owner_account_id'] != account_id), 'reviewers'].values
        rev_exists = [account_id in reviewers_list for reviewers_list in reviewers]
        return sum(rev_exists)

    def count_src_trgt_co_changed(self, row):
        """Count how many times source and target were co-changed"""
        # logger.info(self.df_dependent_changes['Source_repo'].value_counts())
        # logger.info(row['Source_repo'].value_counts())
        return len(self.df_dependent_changes[
            (self.df_dependent_changes['Source_repo'] == row['Source_repo']) &
            (self.df_dependent_changes['Target_repo'] == row['Target_repo']) &
            (self.df_dependent_changes['Target_date'] < row['Target_date'])
        ])
    
    def is_pair_cross(self, row):
        if row['related'] is True and (row['project_source'] != row['project_target']):
            return 1
        return 0

    def count_changed_files_overlap(self, row):
        """Calculate the overlap in changed files between source and target"""
        files1 = set(self.changed_files[row['Source']])
        files2 = set(self.changed_files[row['Target']])
        common_files = files1.intersection(files2)
        return len(files1.union(files2)) / len(common_files) if len(common_files) > 0 else 0

    def assign_past_changes(self, row, X):
        """Assign past changes to a target change"""
        thirty_days_ago = row['created'] - timedelta(days=15)
        source_changes = X.loc[
            (X['project'] != row['project']) &
            (X['created'] < row['created']) &
            (X['created'] >= thirty_days_ago),
            'Target'
        ].tail(30).tolist()
        
        source_changes += self.df_dependent_changes.loc[
            (self.df_dependent_changes['is_cross']==True)& 
            (self.df_dependent_changes['Target']==row['Target']), 
            'Source'].tolist()
        return set(source_changes)
    
    def build_pairs_metrics(self, label, target):
        """Build pairs of metrics for a specific fold"""
        logger.info(f'******** Started building pairs for fold {target} ********')
        path = osp.join('.', 'Files', 'Data', label, target)
        X = pd.read_csv(path)
        additional_attrs = ['number', 'created', 'project', 'owner_account_id']

        # X = pd.merge(
        #     left=X, 
        #     right=self.df[], 
        #     left_on='Source', 
        #     right_on='number', 
        #     how='left',
        #     suffixes=('_source', '_target')
        # )

        # Merge with Source data
        X = pd.merge(
            left=X,
            right=self.df[self.METRICS + additional_attrs],
            left_on='Source',
            right_on='number',
            how='inner',
            suffixes=('', '_source')
        )
        # Drop the redundant 'number' column from the right DataFrame
        X.drop(columns=['number'], inplace=True)

        # Merge with Target data
        X = pd.merge(
            left=X,
            right=self.df[self.METRICS + additional_attrs],
            left_on='Target',
            right_on='number',
            how='inner',
            suffixes=('', '_target')
        )
        X.drop(columns=['number'], inplace=True)

         # Rename columns for clarity
        X.rename(columns={
            "owner_account_id_source": "Source_author",
            "owner_account_id_target": "Target_author",
            "created_source": "Source_date",
            "created_target": "Target_date"
        }, inplace=True)

        # Calculate various metrics
        X['changed_files_overlap'] = X.apply(self.count_changed_files_overlap, axis=1)
        logger.info(f'** changed_files_overlap for {target} generated **')

        X['cmn_dev_pctg'] = X.apply(self.compute_common_dev_pctg, axis=1)
        logger.info(f'** cmn_dev_pctg for {target} generated **')

        X['num_shrd_file_tkns'] = X[['Source', 'Target']].apply(
            clas_util.compute_filenames_shared_tokens, args=(self.changed_files,), axis=1
        )
        logger.info(f'** num_shrd_file_tkns for {target} generated **')
        
        X['num_shrd_desc_tkns'] = X[['Source', 'Target']].apply(
            clas_util.compute_shared_desc_tokens, args=(self.changes_description,), axis=1
        )
        logger.info(f'** num_shrd_desc_tkns for {target} generated **')
        
        X['dev_in_src_change_nbr'] = X.apply(self.count_dev_in_src_change, axis=1)
        logger.info(f'** dev_in_src_change_nbr for {target} generated **')
        
        # X['rev_in_src_change_nbr'] = X.apply(self.count_rev_in_src_change, axis=1)
        # logger.info(f'** rev_in_src_change_nbr for {target} generated **')
        
        X['src_trgt_co_changed_nbr'] = X.apply(self.count_src_trgt_co_changed, axis=1)
        logger.info(f'** src_trgt_co_changed_nbr for {target} generated **')

        X['is_cross'] = X.apply(self.is_pair_cross, axis=1)
        logger.info(f'** is_cross for {target} generated **')

        # Drop unnecessary columns
        X.drop(columns=[
            'number_source', 'number_target', #'reviewers', 
            'project_source', 'project_target'
            # 'owner_account_id_source', 'owner_account_id_target'
        ], axis=1, inplace=True, errors='ignore')

        desc_model = clas_util.doc2vec_model(self.df_changes, X[['Source', 'Target']].values, target[-1])
        subject_model = clas_util.doc2vec_model(self.df_changes, X[['Source', 'Target']].values, target[-1], "subject")

        X = clas_util.compute_embdedding_similarity(self.df_changes, desc_model, X, 'commit_message', 'desc')
        X = clas_util.compute_embdedding_similarity(self.df_changes, subject_model, X, 'subject', 'subject')

        # Save the results
        X.to_csv(path, index=None)
        logger.info(f'** {target} successfully saved to the {path} **')

        return target
    

    def build_temp_metric(self, label,target):
        logger.info(f'******** Started building pairs for fold {target} ********')
        path = osp.join('.', 'Files', 'Data', label, target)
        X = pd.read_csv(path)
        X['Source_date'] = pd.to_datetime(X['Source_date'])
        X['Target_date'] = pd.to_datetime(X['Target_date'])
        additional_attrs = ['number', 'project']

        X = pd.merge(
            left=X, 
            right=self.df[additional_attrs], 
            left_on='Source', 
            right_on='number', 
            how='left',
            suffixes=('_target', '_source')
        )
        # # Drop the redundant 'number' column from the right DataFrame
        # X.drop(columns=['number'], inplace=True)
        X = pd.merge(
            left=X, 
            right=self.df[additional_attrs], 
            left_on='Target', 
            right_on='number', 
            how='left',
            suffixes=('_source', '_target')
        )
        # # Drop the redundant 'number' column from the right DataFrame
        X.drop(columns=['number_source', 'number_target'], inplace=True)
 


         # Rename columns for clarity
        X.rename(columns={
            # "owner_account_id_source": "Source_author",
            # "owner_account_id_target": "Target_author",
            # "created_source": "Source_date",
            # "created_target": "Target_date"
            "project_source": "Source_repo",
            "project_target": "Target_repo",
        }, inplace=True)
        
        # logger.info(X.columns[:30])
        X['src_trgt_co_changed_nbr'] = X.apply(self.count_src_trgt_co_changed, axis=1)
        logger.info(f'** src_trgt_co_changed_nbr for {target} generated **')

        

        # Save the results
        X.to_csv(path, index=None)
        logger.info(f'** {target} successfully saved to the {path} **')

        return target

        
    def process_all_folds(self, label="Test"):
        """Process all folds for a given label"""
        fold_files = [f for f in hpr.list_file(osp.join('.', 'Files', 'Data', label))]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # results = [executor.submit(self.build_pairs_metrics, label, cpc) for cpc in fold_files]
            results = [executor.submit(self.build_temp_metric, label, cpc) for cpc in fold_files]

            for out in concurrent.futures.as_completed(results):
                logger.info(f'Features for target-based pair {out.result()} saved to memory successfully')
                
    



def main():
    logger.info(f"Script {__file__} started...")
    
    start_date, start_header = hpr.generate_date("This script started at")

    # Initialize and run the analyzer
    generator = PairMetricsGenerator()
    generator.initialize_global_vars()
    generator.process_all_folds()
    
    end_date, end_header = hpr.generate_date("This script ended at")

    logger.info(start_header)
    logger.info(end_header)
    hpr.diff_dates(start_date, end_date)

    logger.info(f"Script {__file__} ended\n")


if __name__ == '__main__':
    main()