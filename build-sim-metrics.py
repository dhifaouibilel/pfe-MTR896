import os
import os.path as osp
import pandas as pd
import numpy as np
import math
import random
import utils.classifier_util as clas_util
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from utils import helpers as hpr
import concurrent.futures

from logging_config import get_logger

df_changes = None
df_changed_lines = None

logger = get_logger()

def process_folds(fold):
    logger.info(f"Strated processing fold {fold}")
    X_train = pd.read_csv(osp.join(".", "Files", "Data", "Train", f"{fold}.csv"))
    X_train.fillna(0, inplace=True)

    desc_model = clas_util.doc2vec_model(df_changes, X_train[['Source', 'Target']].values, fold)
    subject_model = clas_util.doc2vec_model(df_changes, X_train[['Source', 'Target']].values, fold, "subject")
    add_lines_model = clas_util.doc2vec_model(df_changed_lines, X_train[['Source', 'Target']].values, fold, 'added_lines')
    del_lines_model = clas_util.doc2vec_model(df_changed_lines, X_train[['Source', 'Target']].values, fold, 'deleted_lines')

    path = osp.join(".", "Files", "Data", "Test", f"temp_{fold}.csv")
    X_test = pd.read_csv(path)

    X_test = clas_util.compute_embdedding_similarity(df_changes, desc_model, X_test, 'commit_message', 'desc')
    X_test = clas_util.compute_embdedding_similarity(df_changes, subject_model, X_test, 'subject', 'subject')
    X_test = clas_util.compute_embdedding_similarity(df_changed_lines, add_lines_model, X_test, 'added_lines', 'add_lines')
    X_test = clas_util.compute_embdedding_similarity(df_changed_lines, del_lines_model, X_test, 'deleted_lines', 'del_lines')
    
    X_test.to_csv(path, index=None)

    return f"Fold{fold} processed successfully!"

def init_global_vars():
    df_changes_loc = hpr.combine_openstack_data(changes_path="/Changes3/")
    df_changes_loc = df_changes_loc[(df_changes_loc['status']!='NEW')]

    df_changed_lines_loc = pd.read_csv(osp.join(".", "Files", "changed_lines.csv"))

    global df_changes
    df_changes = df_changes_loc
   
    global df_changed_lines
    df_changed_lines = df_changed_lines_loc


if __name__ == '__main__':
    
    logger.info(f"Script {__file__} started...")
    
    start_date, start_header = hpr.generate_date("This script started at")

    init_global_vars()

    tscv = TimeSeriesSplit(n_splits = 10)
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_folds, fold) for fold in range(0, 10)]

        for out in concurrent.futures.as_completed(results):
            logger.info(out.result())
    
    end_date, end_header = hpr.generate_date("This script ended at")

    logger.info(start_header)

    logger.info(end_header)

    hpr.diff_dates(start_date, end_date)

    logger.info(f"Script {__file__} ended\n")
