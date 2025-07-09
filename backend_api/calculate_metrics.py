import sys
import os
import os.path as osp
import pandas as pd
import numpy as np
import ast
import re
from datetime import timedelta
from typing import List, Dict, Set, Optional, Union


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.db_manager import MongoManager
from utils import constants
import utils.helpers as hpr
from logging_config import get_logger



class MetricService:
    def __init__(self):
        self.mongo = MongoManager()
        self.metrics_collection = 'metrics'
        self.df_changes: Optional[pd.DataFrame] = None
        self.df_deps = self.mongo.read_all("dependencies2")
        self.DESCRIPTION_METRICS = constants.DESCRIPTION
        self.metrics = {}
        self.logger = get_logger()

        if not self.df_deps.empty:
            self.df_deps['Source_date'] = pd.to_datetime(self.df_deps['Source_date'], errors='coerce')
            self.df_deps['Target_date'] = pd.to_datetime(self.df_deps['Target_date'], errors='coerce')
            self.dependent_changes = set(hpr.flatten_list(self.df_deps[['Source', 'Target']].values))
            self.cross_pro_changes = set(hpr.flatten_list(
                self.df_deps[self.df_deps['is_cross'] == 'Cross'][['Source', 'Target']].values))
        else:
            self.dependent_changes = set()
            self.cross_pro_changes = set()


            # Project-related metrics
    # def count_project_age(self, change: pd.Series) -> float:
    #     """Calculate the age of the project when the change was created."""
    #     if self.df_changes is None or self.df_changes.empty:
    #         raise ValueError("df_changes est vide ou non initialisé.")
    #     if self.df_changes['created'].dtype == 'O':  
    #         self.df_changes['created'] = pd.to_datetime(self.df_changes['created'])
    #     project_creation_date = self.df_changes.loc[
    #         self.df_changes['project'] == change['project'], 'created'].iloc[0]
    #     return (change['created'] - project_creation_date).total_seconds()
    def count_project_age(self, change: pd.Series) -> float:
        """Calcule l'âge du projet en secondes au moment du changement."""
        if self.df_changes is None or self.df_changes.empty:
            raise ValueError("df_changes est vide ou non initialisé.")

        # Forcer la conversion de la colonne 'created' en datetime si nécessaire
        if self.df_changes['created'].dtype == 'O':
            self.df_changes['created'] = pd.to_datetime(self.df_changes['created'], errors='coerce')

        # Filtrer les changements du même projet
        project_changes = self.df_changes[self.df_changes['project'] == change['project']]

        # Vérifier qu'on a bien des changements pour ce projet
        if project_changes.empty:
            return 0.0

        # Trier les changements par date de création
        project_changes = project_changes.sort_values(by='created')

        # Prendre la première date connue pour le projet
        project_creation_date = project_changes['created'].iloc[0]

        # Calculer l'âge du projet en secondes à la date du changement
        age_seconds = (change['created'] - project_creation_date).total_seconds()

        return max(age_seconds, 0.0)  # Toujours retourner une valeur >= 0

    
    # Developer-related metrics
    def count_project_changes_owner(self, change: pd.Series) -> int:
        """Count changes by the same owner in the same project before this change."""
        if self.df_changes.empty:
            return 0
        
        return self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['owner_account_id'] == change['owner_account_id']) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_whole_changes_owner(self, change: pd.Series) -> int:
        """Count all changes by the same owner before this change."""
        return self.df_changes.loc[
            (self.df_changes['owner_account_id'] == change['owner_account_id']) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_projects_contributed(self, change: pd.Series) -> int:
        """Count number of projects the owner has contributed to before this change."""
        return self.df_changes.loc[
            (self.df_changes['owner_account_id'] == change['owner_account_id']) &
            (self.df_changes['created'] < change['created']),
            'project'
        ].nunique()
    
    def count_cross_project_changes(self, change: pd.Series) -> int:
        """Count cross-project changes in the same project before this change."""
        return self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['is_cross'] == True) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_within_project_changes(self, change: pd.Series) -> int:
        """Count within-project dependent changes before this change."""
        return self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['is_cross'] == False) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_whole_within_project_changes(self, change: pd.Series) -> int:
        """Count all within-project dependent changes across projects before this change."""
        return self.df_changes.loc[
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['is_cross'] == False) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_cross_pro_changes_owner(self, change: pd.Series) -> int:
        """Count cross-project changes by the same owner in the same project."""
        return self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['is_cross'] == True) &
            (self.df_changes['owner_account_id'] == change['owner_account_id']) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_within_pro_changes_owner(self, change: pd.Series) -> int:
        """Count within-project changes by the same owner in the same project."""
        return self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['is_dependent'] == True) &
            (self.df_changes['is_cross'] == False) &
            (self.df_changes['owner_account_id'] == change['owner_account_id']) &
            (self.df_changes['created'] < change['created']),
            'number'
        ].nunique()
    
    def count_last_x_days_dependent_projects(self, change: pd.Series, days: int) -> int:
        """Count dependent projects in the last X days."""
        days_ago = change['created'] - timedelta(days=days)
        projects = set(hpr.flatten_list(self.df_deps.loc[
            (
                ((self.df_deps['Source_repo'] == change['project']) &
                (self.df_deps['Source_date'] >= days_ago) &
                (self.df_deps['Source_date'] < change['created'])) |
                (self.df_deps['Target_repo'] == change['project']) &
                (self.df_deps['Target_date'] >= days_ago) &
                (self.df_deps['Target_date'] < change['created'])
            ) &
            (self.df_deps['is_cross'] == 'Cross'),
            ['Target_repo', 'Source_repo']
        ].values))
        return len(projects) - 1 if projects else 0
      
    def count_last_x_days_cross_project_changes(self, change: pd.Series, days: int) -> int:
        """Count cross-project changes in the last X days."""
        days_ago = change['created'] - timedelta(days=days)
        return self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['is_cross'] == True) &
            (self.df_changes['created'] >= days_ago) &
            (self.df_changes['created'] < change['created']),
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
    
    
    # Developer-file interaction metrics
    def count_num_dev_modified_files(self, change: pd.Series) -> int:
        """Count number of developers who modified the same files."""
        res = self.df_changes.loc[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['created'] <= change['created']), 
            ["changed_files", "owner_account_id"]
        ]
        res = res.explode('changed_files')
        res = res[res['changed_files'].isin(change['changed_files'])]
        return res['owner_account_id'].nunique()
    
    def count_avg_num_dev_modified_files(self, change: pd.Series) -> float:
        """Calculate average number of developers per modified file."""
        res = self.df_changes[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['created'] <= change['created'])
        ]
        res = res.explode('changed_files')
        res = res[res['changed_files'].isin(change['changed_files'])]
        num_devs = res['owner_account_id'].nunique()
        return num_devs / len(change['changed_files']) if change['changed_files'] else 0
    
    def count_ratio_dep_chan_owner(self, change: pd.Series) -> float:
        """Calculate ratio of dependent changes by owner in project vs all projects."""
        owner_pro_cha = self.df_changes.loc[
           (self.df_changes['project'] == change['project']) &
           (self.df_changes['owner_account_id'] == change['owner_account_id']) &
           (self.df_changes['is_dependent'] == True) &
           (self.df_changes['created'] <= change['created']),
           "number"
        ].nunique()
        
        owner_all_cha = self.df_changes.loc[
           (self.df_changes['owner_account_id'] == change['owner_account_id']) &
           (self.df_changes['is_dependent'] == True) &
           (self.df_changes['created'] <= change['created']),
           "number"
        ].nunique()
        
        return 100 * (owner_pro_cha / owner_all_cha) if owner_all_cha else 0
      
    def count_num_file_changes(self, change: pd.Series) -> int:
        """Count number of changes that modified the same files."""
        res = self.df_changes[
            (self.df_changes['project'] == change['project']) &
            (self.df_changes['created'] <= change['created'])
        ]
        res = res.explode('changed_files')
        return res.loc[res['changed_files'].isin(change['changed_files']), 'number'].nunique()
    def compute_description_metrics(self, change: pd.Series):
        """
        Calcule les métriques sémantiques à partir du message de commit.
        """
        desc = change.get('commit_message', '').lower()
        desc = re.sub(r'\s+', ' ', desc)  # Nettoyage des espaces multiples

        for attr, rules in self.DESCRIPTION_METRICS.items():
            inclusion_words = rules.get('inclusion', [])
            pattern = rules.get('pattern', None)

            # Vérifie si un mot complet ou une phrase exacte est présente dans la description
            has_keyword = any(
                re.search(rf'\b{re.escape(word)}\b', desc) if ' ' not in word else word in desc
                for word in inclusion_words
            )

            # Vérifie si le pattern regex est présent
            matches_pattern = bool(re.search(pattern, desc)) if pattern else False

            self.metrics[attr] = int(has_keyword or matches_pattern)
            
    def identify_desc_nature(self, desc: str, keyword: str) -> int:
        """Identify if description contains specific keywords."""
        words = re.sub(r'\s+', ' ', desc.lower()).split()
        change_type = self.DESCRIPTION_METRICS[keyword]
        return int(any(
            word for word in words
            if word in change_type['inclusion'] or
            (change_type['pattern'] and re.search(change_type['pattern'], word))
        ))
    # def compute_description_metrics(self, change: pd.Series):
    #     """
    #     Calcule les métriques sémantiques à partir du message de commit.
    #     """
    #     desc = change.get('commit_message', '').lower()

    #     for attr, rules in self.DESCRIPTION_METRICS.items():
    #         inclusion_words = rules.get('inclusion', [])
    #         pattern = rules.get('pattern', None)

    #         # Check inclusion (word or phrase)
    #         has_keyword = any(
    #             re.search(rf'\b{re.escape(word)}\b', desc) if ' ' not in word else word in desc
    #             for word in inclusion_words
    #         )

    #         # Check regex pattern
    #         matches_pattern = bool(re.search(pattern, desc)) if pattern else False

    #         self.metrics[attr] = int(has_keyword or matches_pattern)
        
    # def compute_description_metrics(self, change: pd.Series) -> pd.Series:
    #     """
    #     Calcule les métriques sémantiques à partir du message de commit.
    #     """
    #     desc = change.get('commit_message', '').lower()
    #     for attr, rules in self.DESCRIPTION_METRICS.items():
    #         inclusion_words = rules['inclusion']
    #         pattern = rules.get('pattern')

    #         has_keyword = any(word in desc for word in inclusion_words)
    #         matches_pattern = bool(re.search(pattern, desc)) if pattern else False

    #         self.metrics[attr] = int(has_keyword or matches_pattern)

        
        
    def generate_metrics_for_change(self, change_dict: Dict) -> Dict:
        try:
            change = pd.Series(change_dict)
            change['created'] = pd.to_datetime(change['created'], errors='coerce')

            if isinstance(change.get("changed_files"), str):
                try:
                    change["changed_files"] = ast.literal_eval(change["changed_files"])
                except Exception:
                    change["changed_files"] = []

            self.df_changes = self.mongo.read_filtered_changes()
            self.df_changes["is_dependent"] = self.df_changes["number"].map(lambda x: x in self.dependent_changes)
            self.df_changes["is_cross"] = self.df_changes["number"].map(lambda x: x in self.cross_pro_changes)

            self.metrics['number'] = change['number'] 
            
            # Changes metrics
            self.metrics['insertions'] = change.get('insertions', 0)
            self.metrics['deletions'] = change.get('deletions', 0)
            self.metrics['code_churn'] = change.get('insertions', 0) + change.get('deletions', 0)
            # self.metrics['project_changes_count'] = self.count_project_changes(change)
            # self.metrics['whole_changes_count'] = self.count_whole_changes(change)
            # self.metrics['projects_changes_deps'] = self.count_projects_changes_deps(change)
            # self.metrics['whole_changes_deps'] = self.count_whole_changes_deps(change)
            
            #Project metrics
            self.metrics['project_age'] = self.count_project_age(change)
            self.metrics['cross_project_changes'] = self.count_cross_project_changes(change)
            self.metrics['within_project_changes'] = self.count_within_project_changes(change)
            self.metrics['whole_within_project_changes'] = self.count_whole_within_project_changes(change)
            self.metrics['last_mth_dep_proj_nbr'] = self.count_last_x_days_dependent_projects(change, days=30)
            # self.metrics['avg_cro_proj_nbr'] = self.count_avg_cro_proj_nbr(change)
            self.metrics['last_mth_cro_proj_nbr'] = self.count_last_x_days_cross_project_changes(change, days=30)
            # self.metrics['last_mth_mod_uniq_proj_nbr'] = self.count_last_x_days_modified_unique_projects(change, days=30)

            # Developer metrics
            self.metrics['project_changes_owner'] = self.count_project_changes_owner(change)
            self.metrics['whole_changes_owner'] = self.count_whole_changes_owner(change)
            self.metrics['projects_contributed_owner'] = self.count_projects_contributed(change)
            
            self.metrics['cross_project_changes_owner'] = self.count_cross_pro_changes_owner(change)
            self.metrics['within_project_changes_owner'] = self.count_within_pro_changes_owner(change)
            self.metrics['ratio_dep_chan_owner'] = self.count_ratio_dep_chan_owner(change)
            
            

            # File metrics
            self.metrics['num_file_types'] = self.count_num_file_types(change['changed_files'])
            self.metrics['num_directory_files'] = self.count_num_directory_files(change['changed_files'])
            self.metrics['num_dev_modified_files'] = self.count_num_dev_modified_files(change)
            self.metrics['avg_num_dev_modified_files'] = self.count_avg_num_dev_modified_files(change)
            self.metrics['num_file_changes'] = self.count_num_file_changes(change)
            # self.metrics['num_merged_changes'] = self.count_type_changes(change, 'MERGED')
            # self.metrics['num_abandoned_changes'] = self.count_type_changes(change, 'ABANDONED')
            
            #Text metrics
            self.metrics['subject_length'] = len(change.get('subject', ''))
            self.metrics['description_length'] = self.count_desc_length(change.get('commit_message', ''))
            # self.metrics['subject_word_count'] = self.count_words(change.get('subject', ''))
            self.metrics['description_word_count'] = self.count_words(change.get('commit_message', ''))
            # Extraire la valeur pour le changement courant uniquement
            matched = self.df_changes[self.df_changes["number"] == change["number"]]["is_dependent"]
            self.metrics["is_dependent"] = int(matched.values[0]) if not matched.empty else 0
            # self.metrics['is_dependent'] = self.df_changes['is_dependent']
            
            # Description metrics
            # self.compute_description_metrics(change)
            for attr in self.DESCRIPTION_METRICS:
                self.metrics[attr] = self.identify_desc_nature(change['commit_message'], attr)
            
            

            enriched_doc = {k: v for k, v in self.metrics.items() if v is not None}
            self.mongo.insert_data("metrics", enriched_doc)
            return enriched_doc
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques pour le changement {change_dict.get('number')}: {e}", exc_info=True)
            raise

