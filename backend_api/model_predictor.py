import pandas as pd

class ModelPredictor:
    def __init__(self, model, model_type: str = "m1"):
        self.model = model
        self.model_type = model_type.lower()

        # Définir les features selon le type de modèle
        if self.model_type == "m1":
            self.features = [
                'cross_project_changes', 'num_file_types', 'is_preventive',
                'last_mth_cro_proj_nbr', 'project_changes_owner', 'whole_changes_owner',
                'is_refactoring', 'cross_project_changes_owner',
                'whole_within_project_changes', 'is_merge', 'num_directory_files',
                'within_project_changes', 'is_corrective', 'num_file_changes',
                'project_age', 'is_non_functional', 'subject_length',
                'within_project_changes_owner', 'avg_num_dev_modified_files',
                'ratio_dep_chan_owner', 'projects_contributed_owner'
            ]
        elif self.model_type == "m2":
            self.features = [
                'changed_files_overlap', 'cmn_dev_pctg', 'num_shrd_file_tkns','num_shrd_desc_tkns', 'dev_in_src_change_nbr',
                'src_trgt_co_changed_nbr', 'has_feature_addition_source',
                'deletions_source', 'cross_project_changes_source', 'insertions_source',
                'num_file_types_source', 'is_preventive_source',
                'last_mth_cro_proj_nbr_source', 'project_changes_owner_source',
                'whole_changes_owner_source', 'is_refactoring_source',
                'cross_project_changes_owner_source', 'description_word_count_source',
                'whole_within_project_changes_source', 'is_merge_source',
                'num_directory_files_source', 'within_project_changes_source',
                'is_corrective_source', 'description_length_source',
                'num_file_changes_source', 'project_age_source',
                'is_non_functional_source', 'subject_length_source',
                'within_project_changes_owner_source', 'num_dev_modified_files_source',
                'avg_num_dev_modified_files_source', 'code_churn_source',
                'ratio_dep_chan_owner_source', 'projects_contributed_owner_source',
                'has_feature_addition_target', 'deletions_target',
                'cross_project_changes_target', 'insertions_target',
                'num_file_types_target', 'is_preventive_target',
                'last_mth_cro_proj_nbr_target', 'project_changes_owner_target',
                'whole_changes_owner_target', 'is_refactoring_target',
                'cross_project_changes_owner_target', 'description_word_count_target',
                'whole_within_project_changes_target', 'is_merge_target',
                'num_directory_files_target', 'within_project_changes_target',
                'is_corrective_target', 'description_length_target',
                'num_file_changes_target', 'project_age_target',
                'is_non_functional_target', 'subject_length_target',
                'within_project_changes_owner_target', 'num_dev_modified_files_target',
                'avg_num_dev_modified_files_target', 'code_churn_target',
                'ratio_dep_chan_owner_target', 'projects_contributed_owner_target',
                'desc_sim', 'subject_sim'
            ]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def filter_predicted_pairs(self, possible_deps_numbers, predictions):
        return [
        pair for pair, pred in zip(possible_deps_numbers, predictions)
        if pred == 1
    ]
        
    def predict(self, features: dict | pd.DataFrame) -> int | list:
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise ValueError("Input must be a dict or a DataFrame.")

        # Reordonner les colonnes selon l'ordre attendu
        df = df.reindex(columns=self.features)

        # Vérifier les colonnes manquantes
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            raise ValueError(f"Colonnes manquantes ou mal nommées dans l'entrée : {missing_cols}")

        # Prédiction
        result = self.model.predict(df)

        if self.model_type == "m1":
            return result[0]  # Une seule prédiction
        else:  # m2 : prédiction d'une liste
            return result.tolist()


# features_m1 = [
#     'cross_project_changes', 'num_file_types', 'is_preventive',
#     'last_mth_cro_proj_nbr', 'project_changes_owner', 'whole_changes_owner',
#     'is_refactoring', 'cross_project_changes_owner',
#     'whole_within_project_changes', 'is_merge', 'num_directory_files',
#     'within_project_changes', 'is_corrective', 'num_file_changes',
#     'project_age', 'is_non_functional', 'subject_length',
#     'within_project_changes_owner', 'avg_num_dev_modified_files',
#     'ratio_dep_chan_owner', 'projects_contributed_owner'
# ]

# features_m2 = [
#         'changed_files_overlap', 'cmn_dev_pctg', 'num_shrd_file_tkns','num_shrd_desc_tkns', 'dev_in_src_change_nbr',
#        'src_trgt_co_changed_nbr', 'has_feature_addition_source',
#        'deletions_source', 'cross_project_changes_source', 'insertions_source',
#        'num_file_types_source', 'is_preventive_source',
#        'last_mth_cro_proj_nbr_source', 'project_changes_owner_source',
#        'whole_changes_owner_source', 'is_refactoring_source',
#        'cross_project_changes_owner_source', 'description_word_count_source',
#        'whole_within_project_changes_source', 'is_merge_source',
#        'num_directory_files_source', 'within_project_changes_source',
#        'is_corrective_source', 'description_length_source',
#        'num_file_changes_source', 'project_age_source',
#        'is_non_functional_source', 'subject_length_source',
#        'within_project_changes_owner_source', 'num_dev_modified_files_source',
#        'avg_num_dev_modified_files_source', 'code_churn_source',
#        'ratio_dep_chan_owner_source', 'projects_contributed_owner_source',
#        'has_feature_addition_target', 'deletions_target',
#        'cross_project_changes_target', 'insertions_target',
#        'num_file_types_target', 'is_preventive_target',
#        'last_mth_cro_proj_nbr_target', 'project_changes_owner_target',
#        'whole_changes_owner_target', 'is_refactoring_target',
#        'cross_project_changes_owner_target', 'description_word_count_target',
#        'whole_within_project_changes_target', 'is_merge_target',
#        'num_directory_files_target', 'within_project_changes_target',
#        'is_corrective_target', 'description_length_target',
#        'num_file_changes_target', 'project_age_target',
#        'is_non_functional_target', 'subject_length_target',
#        'within_project_changes_owner_target', 'num_dev_modified_files_target',
#        'avg_num_dev_modified_files_target', 'code_churn_target',
#        'ratio_dep_chan_owner_target', 'projects_contributed_owner_target',
#        'desc_sim', 'subject_sim'
#        ]

# def predict_dependencies_m1(model, metrics: dict):
#     # Convertir le dictionnaire en DataFrame (ligne unique)
#     df = pd.DataFrame([metrics])
    
#      # S'assurer que les colonnes sont bien dans l'ordre attendu
#     df = df.reindex(columns=features_m1)

#     # Vérifier les colonnes manquantes
#     missing_cols = df.columns[df.isnull().any()].tolist()
#     if missing_cols:
#         raise ValueError(f"Colonnes manquantes ou mal nommées dans l'entrée : {missing_cols}")


#     prediction = model.predict(df)
#     return prediction[0]

# def predict_pairs_m2(model, pair_metrics: pd.DataFrame):
#     # Convertir le dictionnaire en DataFrame (ligne unique)
#     df = pd.DataFrame([pair_metrics])
    
#      # S'assurer que les colonnes sont bien dans l'ordre attendu
#     df = df.reindex(columns=features_m2)

#     # Vérifier les colonnes manquantes
#     missing_cols = df.columns[df.isnull().any()].tolist()
#     if missing_cols:
#         raise ValueError(f"Colonnes manquantes ou mal nommées dans l'entrée : {missing_cols}")
#     preds = model.predict(df)
#     return preds.tolist()