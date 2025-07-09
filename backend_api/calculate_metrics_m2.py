import sys
import os
import os.path as osp
import pandas as pd
import numpy as np
import concurrent.futures
import ast
import re
from datetime import timedelta
from typing import List, Dict, Set, Optional, Union


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_manager import MongoManager
from utils import constants, classifier_util as clas_util
from gensim.models.doc2vec import Doc2Vec
import utils.helpers as hpr
from logging_config import get_logger



class PairMetricsService:
    def __init__(self):
        self.METRICS = [m for m in constants.get_metrics()[:-8]]
        self.df_dependent_changes = None
        self.dependent_changes = None
        self.cross_pro_changes = None
        self.within_pro_changes = None
        self.df = None
        self.changed_files = None
        self.changes_description = None
        self.added_lines = None
        self.deleted_lines = None
        self.changes_collection = 'changes'
        self.metrics_collection = 'metrics'
        self.deps_collection = 'dependencies2'
        self.mongo_manager = MongoManager()
        self.logger = get_logger()
        self.initialize_global_vars()


        # if not self.df_deps.empty:
        #     self.df_deps['Source_date'] = pd.to_datetime(self.df_deps['Source_date'], errors='coerce')
        #     self.df_deps['Target_date'] = pd.to_datetime(self.df_deps['Target_date'], errors='coerce')
        #     self.dependent_changes = set(hpr.flatten_list(self.df_deps[['Source', 'Target']].values))
        #     self.cross_pro_changes = set(hpr.flatten_list(
        #         self.df_deps[self.df_deps['is_cross'] == 'Cross'][['Source', 'Target']].values))
        # else:
        #     self.dependent_changes = set()
        #     self.cross_pro_changes = set()
            
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

        # # Process OpenStack data
        # # df_changes = hpr.combine_openstack_data(changes_path="/Changes3/")
        # df_changes = self.mongo_manager.read_filtered_changes()

        # def safe_literal_eval(x):
        #     if isinstance(x, str):
        #         try:
        #             return ast.literal_eval(x)
        #         except (ValueError, SyntaxError):
        #             return None
        #     return x  # déjà un objet Python
        # self.logger.info(df_changes.columns)
        # df_changes['reviewers'] = df_changes['reviewers'].map(safe_literal_eval)
        # # df_changes['reviewers'] = df_changes['reviewers'].map(ast.literal_eval)
        # df_changes['reviewers'] = df_changes['reviewers'].map( lambda x: [rev['_account_id'] for rev in x] if isinstance(x, list) else [])
        # # df_changes['reviewers'] = df_changes['reviewers'].map(lambda x: [rev['_account_id'] for rev in x])
        # df_changes['changed_files'] = df_changes['changed_files'].map(hpr.combine_changed_file_names)
        # df_changes['commit_message'] = df_changes['commit_message'].map(hpr.preprocess_change_description)

        # # Combine features
        # # df_features = self.combine_features()
        # df_features = self.mongo_manager.read_all(self.metrics_collection)
        # df_features = pd.merge(
        #     left=df_features, 
        #     right=df_changes[['number', 'created', 'project', 'owner_account_id', 'reviewers']], 
        #     left_on='number', 
        #     right_on='number', 
        #     how='inner',
        #     suffixes=('_source', '_target')
        # )

        # df_features['is_dependent'] = df_features['number'].map(lambda nbr: 1 if nbr in dependent_changes_loc else 0)
        # df_features['is_cross'] = df_features['number'].map(self.is_cross_project)

        # # Create dictionaries for quick lookups
        # self.changed_files = dict(zip(df_changes['number'], df_changes['changed_files']))
        # self.changes_description = dict(zip(df_changes['number'], df_changes['commit_message']))
        # self.added_lines = dict(zip(df_changes['number'], df_changes['insertions']))
        # self.deleted_lines = dict(zip(df_changes['number'], df_changes['deletions']))
        
        # del df_changes  # Free up memory

        # self.df = df_features
        
    def prepare_df_features(
        self,
        target_change_number: int,
        # dependent_changes_loc: set,
) -> pd.DataFrame:
        """
        Prépare le contexte nécessaire pour le calcul des métriques de paires.

        Args:
            df_features: DataFrame des métriques principales.
            target_change_number: le numéro de changement à récupérer.

        Returns:
            df_features enrichi
        """
        # Process OpenStack data
        df_changes = self.mongo_manager.read_filtered_changes(num_docs=1000)
        # Vérifie si le changement cible est présent, sinon l’ajoute
        if target_change_number not in df_changes['number'].values:
            self.logger.warning(f"Target change {target_change_number} not in df_changes — fetching individually.")
            target_doc = self.mongo_manager.get_change_by_number(target_change_number)
            if target_doc:
                target_df = pd.DataFrame([target_doc])
                df_changes = pd.concat([df_changes, target_df], ignore_index=True)
            else:
                raise ValueError(f"Target change {target_change_number} introuvable dans MongoDB.")


        def safe_literal_eval(x):
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except (ValueError, SyntaxError):
                    return None
            return x  # déjà un objet Python
        self.logger.info(df_changes.columns)
        df_changes['reviewers'] = df_changes['reviewers'].map(safe_literal_eval)
        # df_changes['reviewers'] = df_changes['reviewers'].map(ast.literal_eval)
        df_changes['reviewers'] = df_changes['reviewers'].map( lambda x: [rev['_account_id'] for rev in x] if isinstance(x, list) else [])
        # df_changes['reviewers'] = df_changes['reviewers'].map(lambda x: [rev['_account_id'] for rev in x])
        df_changes['changed_files'] = df_changes['changed_files'].map(hpr.combine_changed_file_names)
        df_changes['commit_message'] = df_changes['commit_message'].map(hpr.preprocess_change_description)

        # Combine features
        # df_features = self.combine_features()
        df_features = self.mongo_manager.read_all(self.metrics_collection)
        # df_features = pd.merge(
        #     left=df_features, 
        #     right=df_changes[['number', 'created', 'project', 'owner_account_id', 'reviewers']], 
        #     left_on='number', 
        #     right_on='number', 
        #     how='inner',
        #     suffixes=('_source', '_target')
        # )
        

        # Sécuriser reviewers si disponible
        if 'reviewers' in df_changes.columns:
            df_changes['reviewers'] = df_changes['reviewers'].map(safe_literal_eval)

        # # Fusionner les attributs requis
        # df_features = pd.merge(
        #     df_features,
        #     df_changes[['number', 'created', 'project', 'owner_account_id', 'reviewers']],
        #     on='number',
        #     how='inner'
        # )

        df_features['is_dependent'] = df_features['number'].map(lambda nbr: 1 if nbr in self.dependent_changes else 0)
        df_features['is_cross'] = df_features['number'].map(self.is_cross_project)

        # Create dictionaries for quick lookups
        self.changed_files = dict(zip(df_changes['number'], df_changes['changed_files']))
        self.changes_description = dict(zip(df_changes['number'], df_changes['commit_message']))
        self.desc_df = pd.DataFrame([
            {'number': num, 'commit_message': msg}
            for num, msg in self.changes_description.items()
        ])
        self.changes_subject = dict(zip(df_changes['number'], df_changes['subject']))

        self.subject_df = pd.DataFrame([
            {'number': num, 'subject': subj}
            for num, subj in self.changes_subject.items()
        ])
        self.added_lines = dict(zip(df_changes['number'], df_changes['insertions']))
        self.deleted_lines = dict(zip(df_changes['number'], df_changes['deletions']))
        del df_changes  # Free up memory

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
    
    def compute_pctg_cross_project_changes(self, change):
        """Calculate percentage of cross-project changes"""
        dominator = change['cross_project_changes'] + change['within_project_changes']
        if dominator == 0:
            return 0
        return change['cross_project_changes'] / dominator

    def compute_pctg_whole_cross_project_changes(self, change):
        """Calculate percentage of whole cross-project changes"""
        dominator = change['whole_cross_project_changes'] + change['whole_within_project_changes']
        if dominator == 0:
            return 0
        return change['whole_cross_project_changes'] / dominator

    def compute_ptg_cross_project_changes_owner(self, change):
        """Calculate percentage of cross-project changes by owner"""
        dominator = change['cross_project_changes_owner'] + change['within_project_changes_owner']
        if dominator == 0:
            return 0
        return change['cross_project_changes_owner'] / dominator
    
    def is_cross_project(self, number):
        """Determine if a change is cross-project, within-project, or neither"""
        if number in self.cross_pro_changes:
            return 1
        elif number in self.within_pro_changes:
            return 0
        else:
            return 2
    
    def compute_common_dev_pctg(self, change):
        """Calculate percentage of common developers between source and target projects"""
        dev_source = self.df.loc[
            (self.df['project'] == change['Source_repo']) &
            (self.df['created'] < change['Target_date']),
            'owner_account_id'
        ].unique()
        dev_target = self.df.loc[
            (self.df['project'] == change['Target_repo']) &
            (self.df['created'] < change['Target_date']),
            'owner_account_id'
        ].unique()

        union = len(set(dev_source).union(dev_target))
        intersect = len(set(dev_source).intersection(dev_target))
        return intersect/union if union != 0 else 0

    def count_dev_in_src_change(self, change):
        """Count developer's changes in the source project"""
        changes_nbr = self.df.loc[
            (self.df['project'] == change['Source_repo']) &
            (self.df['created'] < change['Target_date']) &
            (self.df['owner_account_id'] == change['Target_author']),
            'number'
        ].nunique()

        return changes_nbr

    def count_rev_in_src_change(self, change):
        """Count reviewer's changes in the source project"""
        account_id = change['Target_author']
        reviewers = self.df.loc[
            (self.df['project'] == change['Source_repo']) &
            (self.df['created'] < change['Target_date']) & 
            (self.df['owner_account_id'] != account_id), 'reviewers'].values
        rev_exists = [account_id in reviewers_list for reviewers_list in reviewers]
        return sum(rev_exists)

    def count_src_trgt_co_changed(self, change):
        """Count how many times source and target were co-changed"""
        # logger.info(self.df_dependent_changes['Source_repo'].value_counts())
        # logger.info(change['Source_repo'].value_counts())
        return len(self.df_dependent_changes[
            (self.df_dependent_changes['Source_repo'] == change['Source_repo']) &
            (self.df_dependent_changes['Target_repo'] == change['Target_repo']) &
            (self.df_dependent_changes['Target_date'] < change['Target_date'])
        ])
    
    # def is_pair_cross(self, change):
    #     if change['related'] is True and (change['Source_repo'] != change['Target_repo']):
    #         return 1
    #     return 0
    
    def is_pair_cross(self, change):
        return int(change['Source_repo'] != change['Target_repo'])

    def count_changed_files_overlap(self, change):
        """Calculate the overlap in changed files between source and target"""
        files1 = set(self.changed_files[change['Source']])
        files2 = set(self.changed_files[change['Target']])
        common_files = files1.intersection(files2)
        return len(files1.union(files2)) / len(common_files) if len(common_files) > 0 else 0

    def assign_past_changes(self, change, X):
        """Assign past changes to a target change"""
        thirty_days_ago = change['created'] - timedelta(days=15)
        source_changes = X.loc[
            (X['project'] != change['project']) &
            (X['created'] < change['created']) &
            (X['created'] >= thirty_days_ago),
            'Target'
        ].tail(30).tolist()
        
        source_changes += self.df_dependent_changes.loc[
            (self.df_dependent_changes['is_cross']==True)& 
            (self.df_dependent_changes['Target']==change['Target']), 
            'Source'].tolist()
        return set(source_changes)
        
        
    def build_pairs_metrics_for_change(self, change_number: int) -> pd.DataFrame:
        """Build dependency pair metrics for a single change against all others."""
        try:
            self.logger.info(f"Generating pair metrics for change #{change_number}")
            self.prepare_df_features(change_number)

            # Récupérer le changement cible
            target_change = self.df[self.df['number'] == change_number].copy()
            # self.logger.info(f"target change is: {target_change}")

            if target_change.empty:
                raise ValueError(f"Change #{change_number} not found in dataframe.")

            target_change = target_change.iloc[0]  # Convertir en Series

            # Préparer le DataFrame source avec tous les autres changements
            source_df = self.df[self.df['number'] != change_number].copy()
            required_cols = ['number', 'created', 'project', 'owner_account_id']
            for col in required_cols:
                if col not in target_change:
                    raise KeyError(f"Missing column '{col}' in target_change: keys = {target_change.keys()}")

            # Ajouter les colonnes du changement cible à tous les autres
            for col in self.METRICS + ['number', 'created', 'project', 'owner_account_id']:
                source_df[f"{col}_target"] = target_change[col]

            # source_df.rename(columns={
            #     "number": "Source",
            #     "created": "created_source",
            #     "project": "project_source",
            #     "owner_account_id": "owner_account_id_source"
            # }, inplace=True)
            # Renommer toutes les colonnes sources avec le suffixe _source, sauf les colonnes nouvellement ajoutées (_target)
            source_df.rename(columns=lambda col: f"{col}_source" if col in (self.METRICS + ['number', 'created', 'project', 'owner_account_id']) else col, inplace=True)

            # Renommer 'number_source' en 'Source' (clé primaire)
            source_df.rename(columns={'number_source': 'Source'}, inplace=True)

            source_df["Target"] = change_number
            source_df["created_target"] = target_change["created"]
            source_df["project_target"] = target_change["project"]
            source_df["owner_account_id_target"] = target_change["owner_account_id"]

            # Renommer pour correspondre aux anciennes conventions
            X = source_df.copy()

            X["Source_date"] = X["created_source"]
            X["Target_date"] = X["created_target"]
            X["Source_author"] = X["owner_account_id_source"]
            X["Target_author"] = X["owner_account_id_target"]
            X.rename(columns={
                # "owner_account_id_source": "Source_author",
                # "owner_account_id_target": "Target_author",
                # "created_source": "Source_date",
                # "created_target": "Target_date"
                # "project_age": "project_age_source",
                "project_source": "Source_repo",
                "project_target": "Target_repo",
            }, inplace=True)

            # Calcul des métriques
            X['changed_files_overlap'] = X.apply(self.count_changed_files_overlap, axis=1)
            X['cmn_dev_pctg'] = X.apply(self.compute_common_dev_pctg, axis=1)
            X['num_shrd_file_tkns'] = X[['Source', 'Target']].apply(
                clas_util.compute_filenames_shared_tokens, args=(self.changed_files,), axis=1
            )
            X['num_shrd_desc_tkns'] = X[['Source', 'Target']].apply(
                clas_util.compute_shared_desc_tokens, args=(self.changes_description,), axis=1
            )
            X['dev_in_src_change_nbr'] = X.apply(self.count_dev_in_src_change, axis=1)
            X['src_trgt_co_changed_nbr'] = X.apply(self.count_src_trgt_co_changed, axis=1)
            X['is_cross'] = X.apply(self.is_pair_cross, axis=1)
            X = clas_util.compute_embdedding_similarity(
                df_changes=self.desc_df,
                model=Doc2Vec,
                X=X,
                attr='commit_message',
                label='desc'
            )
            X = clas_util.compute_embdedding_similarity(
                df_changes=self.subject_df,
                model=Doc2Vec,
                X=X,
                attr='subject',
                label='subject'
            )


            # X['subject_sim'] = X[['Source', 'Target']].apply(
            #     clas_util.compute_shared_desc_tokens, args=(self.changes_description,), axis=1
            # )
            # Supprimer les colonnes inutiles
            X.drop(columns=[
                'created_source', 'created_target', 'owner_account_id_source', 'owner_account_id_target', 'number_target', 'number_source',
                'project_source', 'project_target', 'reviewers',
            ], axis=1, inplace=True, errors='ignore')

            self.logger.info(f"Pair metrics computed for change #{change_number} with {len(X)} pairs.")
            output_path = f"./Files/Results/pairs_metrics_{change_number}.csv"
            X.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"Métriques de paires sauvegardées dans : {output_path}")
            return X
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de pairs  pour le changement {change_number}: {e}", exc_info=True)
            raise

    
      
    def process_all_folds(self, label="Test"):
        """Process all folds for a given label"""
        fold_files = [f for f in hpr.list_file(osp.join('.', 'Files', 'Data', label))]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(self.build_pairs_metrics, label, cpc) for cpc in fold_files]
            

            for out in concurrent.futures.as_completed(results):
                logger.info(f'Features for target-based pair {out.result()} saved to memory successfully')
                
    


def main():
    logger.info(f"Script {__file__} started...")
    
    start_date, start_header = hpr.generate_date("This script started at")

    # Initialize and run the analyzer
    generator = PairMetricsService()
    generator.initialize_global_vars()
    generator.process_all_folds()
    
    end_date, end_header = hpr.generate_date("This script ended at")

    logger.info(start_header)
    logger.info(end_header)
    hpr.diff_dates(start_date, end_date)

    logger.info(f"Script {__file__} ended\n")


if __name__ == '__main__':
    main()