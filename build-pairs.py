import os
import os.path as osp
import pandas as pd
import numpy as np
import math
import random
from datetime import timedelta, datetime
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from utils import helpers as hpr
import concurrent.futures

from logging_config import get_logger

df_changes = None
target_numbers = None
df_dependencies = None

logger = get_logger()


def assign_past_changes(row):
    days_offset = row['created'] - timedelta(days=15)
    source_changes = df_changes.loc[
        (df_changes['created'] < row['created']) &
        (df_changes['created'] >= days_offset),
        'number'
    ].tolist()

    # if len(source_changes) >= 60:
    #     source_changes = random.sample(source_changes, 60)
    
    # source_changes += df_dependencies.loc[
    #     (df_dependencies['Target']==row['Target']), 
    #     'Source'].tolist()
    return set(source_changes)

def build_pairs(target, fold):
    logger.info(f'******** Started building pairs of changes for Fold {fold}')
    X = df_changes.loc[df_changes['number'].isin(target), ['number', 'created']]
    X = X.rename(columns={'number': 'Target'})
    X['Source'] = X.apply(assign_past_changes, axis=1)
    logger.info(f'Source changes assigned Fold {fold}')
    X = X.explode(column='Source')
    logger.info(f'Source changes exploded Fold {fold}')
    X.dropna(subset=['Source'], inplace=True)
    X = pd.merge(
        left=X, 
        right=df_changes[['number', 'created']], 
        left_on=['Source'], 
        right_on=['number'], 
        how='left',
        suffixes=('_target', '_source')
    )
    X.sort_values(by=['created_target', 'created_source'], inplace=True)
    X.reset_index(drop=True, inplace=True)

    X = pd.merge(
        left=X, 
        right=df_dependencies[['Source', 'Target', 'related']], 
        left_on=['Source', 'Target'], 
        right_on=['Source', 'Target'], 
        how='left',
        suffixes=('_target', '_source')
    )

    X['related'].fillna(0, inplace=True)
    X['related'] = X['related'].map(int)

    # X.drop(columns=['number', 'owner_account_id', 'project'], inplace=True)

    return X[['Source', 'Target', 'related']]

def process_folds(fold, train_idx, test_idx):
    train_numbers = target_numbers[train_idx]
    test_numbers = target_numbers[test_idx]

    # df_train = build_pairs(train_numbers, fold)
    logger.info(f"Training set for Fold {fold} has been processed")
    # y_train = df_train['related']
    # df_train = df_train.drop(columns=['related'])

    # ros = RandomUnderSampler(random_state=42)
        
    # Perform under-sampling of the majority class(es)
    # df_train, y_train = ros.fit_resample(df_train, y_train)
    # df_train['related'] = y_train

    # rand_represe_sample = select_representative_sample(list(test_numbers))

    # Set a seed for reproducibility (e.g., 42)
    random.seed(42)

    # Take a random sample of 100 (without replacement)
    rand_represe_sample = random.sample(test_numbers.tolist(), 100)

    df_test = build_pairs(rand_represe_sample, fold)
    logger.info(f"Test set for Fold {fold} has been processed")
    # test_pos = df_test[df_test['related']==1]
    # test_neg = df_test[df_test['related']==0]

    # test_pos = test_pos.sample(n=int(len(test_pos)*.10), random_state=42)
    # test_neg = test_neg.sample(n=int(len(test_neg)*.10), random_state=42)

    # df_test = pd.concat((test_pos, test_neg))

    # df_train.to_csv(osp.join(".", "Files", "Data", "Train", f"temp_{fold}.csv"), index=None)
    df_test.to_csv(osp.join(".", "Files", "Data", "Test", f"temp100_15_{fold}.csv"), index=None)

    return f"Fold{fold} processed successfully!"

def calculate_sample_size(population_size, confidence_level=0.95, margin_of_error=0.05, p=0.5):
    # Z-score for 95% confidence
    Z = 1.96 if confidence_level == 0.95 else 1.64 if confidence_level == 0.90 else 2.58
    e = margin_of_error

    # Initial sample size estimate
    n = (Z**2 * p * (1 - p)) / (e**2)

    # Adjust for finite population
    n_adj = n / (1 + ((n - 1) / population_size))
    return math.ceil(n_adj)

def select_representative_sample(data, seed=42, confidence_level=0.95, margin_of_error=0.05):
    population_size = len(data)
    sample_size = calculate_sample_size(population_size, confidence_level, margin_of_error)
    random.seed(seed)
    return random.sample(data, sample_size)

def init_global_vars():
    df_dependencies_loc = pd.read_csv(osp.join(".", "Files", "source_target_evolution_clean.csv"))
    df_dependencies_loc.dropna(subset=["Source_status", "Target_status"], inplace=True)
    df_dependencies_loc = df_dependencies_loc.loc[(df_dependencies_loc['Source_status']!='NEW')&(df_dependencies_loc['Target_status']!='NEW')]
    df_dependencies_loc['related'] = 1

    df_changes_loc = hpr.combine_openstack_data(changes_path="/Changes3/")
    min_date = datetime(2014, 1, 1)
    df_changes_loc = df_changes_loc[(df_changes_loc['status']!='NEW')&(df_changes_loc['created']>=min_date)]
    df_changes_loc = df_changes_loc.drop_duplicates(subset=['change_id'], keep='last')


    # df_deps_red = df_dependencies_loc[['when_identified']]

    # # Calculate Z-scores
    # z_scores = np.abs((df_deps_red - df_deps_red.mean()) / df_deps_red.std())

    # # Set a threshold for identifying outliers
    # threshold = 3

    # # Filter out the outliers
    # df_clean = df_deps_red[(z_scores < threshold).all(axis=1)]

    # df_dependencies_loc = df_dependencies_loc[df_dependencies_loc.index.isin(df_clean.index)]

    # dependent_changes = set(df_dependencies_loc['Source'].tolist() + df_dependencies_loc['Target'].tolist())
    # df_changes_loc = df_changes_loc[df_changes_loc['number'].isin(dependent_changes)]

    target_numbers_loc = df_dependencies_loc['Target'].unique()

    global df_changes
    df_changes = df_changes_loc

    global target_numbers
    target_numbers = target_numbers_loc

    global df_dependencies
    df_dependencies = df_dependencies_loc


if __name__ == '__main__':
    
    logger.info(f"Script {__file__} started...")
    
    start_date, start_header = hpr.generate_date("This script started at")

    init_global_vars()

    tscv = TimeSeriesSplit(n_splits = 10)
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_folds, fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(tscv.split(target_numbers))]

        for out in concurrent.futures.as_completed(results):
            logger.info(out.result())
    
    end_date, end_header = hpr.generate_date("This script ended at")

    logger.info(start_header)

    logger.info(end_header)

    hpr.diff_dates(start_date, end_date)

    logger.info(f"Script {__file__} ended\n")
