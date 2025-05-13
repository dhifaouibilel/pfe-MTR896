import os
import os.path as osp
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from sklearn.model_selection import TimeSeriesSplit
from imblearn.under_sampling import RandomUnderSampler
from utils import helpers as hpr
import concurrent.futures

df_changes = None
target_numbers = None
df_dependencies = None

def build_cross_project_pairs(file):
    path = osp.join('.', 'Files', 'Data', 'Model3', file)
    df = pd.read_csv(path)
    if (df.empty == False):
        print(f'******** Strated processing {file} ********')
        df.drop(columns=['project', 'owner_account_id', 'number_target', 'number_source', 'number', 'created_target', 'created_source'], axis=1, inplace=True)
        print(f'******** File {file} processed successfully ********')
    #     os.remove(osp.join(path))
    #     print(f'******** File {file} deleted successfully ********')
    # else:
        # df['created_source'] = pd.to_datetime(df['created_source'])
        # df['created_target'] = pd.to_datetime(df['created_target'])
        
        # seven_days_ago = df['created_target'].head(1).tolist()[0] - timedelta(days=7)
        # df = df[df['created_source'] >= seven_days_ago]
        
        # df.to_csv(path, index=None)

    return file

def assign_past_changes(row):
    days_offset = row['created'] - timedelta(days=39)
    source_changes = df_changes.loc[
        (df_changes['created'] < row['created']) &
        (df_changes['created'] >= days_offset),
        'number'
    ].tolist()

    # if len(source_changes) >= 60:
    #     source_changes = random.sample(source_changes, 60)
    
    source_changes += df_dependencies.loc[
        (df_dependencies['Target']==row['Target']), 
        'Source'].tolist()
    return set(source_changes)

def build_pairs(target, fold):
    print(f'******** Started building pairs of changes for Fold {fold}')
    X = df_changes.loc[df_changes['number'].isin(target), ['number', 'created']]
    X = X.rename(columns={'number': 'Target'})
    X['Source'] = X.apply(assign_past_changes, axis=1)
    print(f'Source changes assigned Fold {fold}')
    X = X.explode(column='Source')
    print(f'Source changes exploded Fold {fold}')
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

    df_train = build_pairs(train_numbers, fold)
    print(f"Training set for Fold {fold} has been processed")
    y_train = df_train['related']
    df_train = df_train.drop(columns=['related'])

    ros = RandomUnderSampler(random_state=42)
        
    # Perform under-sampling of the majority class(es)
    df_train, y_train = ros.fit_resample(df_train, y_train)
    df_train['related'] = y_train

    df_test = build_pairs(test_numbers, fold)
    print(f"Test set for Fold {fold} has been processed")
    test_pos = df_test[df_test['related']==1]
    test_neg = df_test[df_test['related']==0]

    test_pos = test_pos.sample(n=int(len(test_pos)*.05), random_state=42)
    test_neg = test_neg.sample(n=int(len(test_neg)*.05), random_state=42)

    df_test = pd.concat((test_pos, test_neg))

    df_train.to_csv(osp.join(".", "Files", "Data", "Train", f"{fold}.csv"))
    df_test.to_csv(osp.join(".", "Files", "Data", "Test", f"{fold}.csv"))

    return f"Fold{fold} processed successfully!"


def init_global_vars():
    df_dependencies_loc = pd.read_csv(osp.join(".", "Files", "source_target_evolution_clean.csv"))
    df_dependencies_loc.dropna(subset=["Source_status", "Target_status"], inplace=True)
    df_dependencies_loc = df_dependencies_loc.loc[(df_dependencies_loc['Source_status']!='NEW')&(df_dependencies_loc['Target_status']!='NEW')]
    df_dependencies_loc['related'] = 1

    df_changes_loc = hpr.combine_openstack_data(changes_path="/Changes3/")
    min_date = datetime(2014, 1, 1)
    df_changes_loc = df_changes_loc[(df_changes_loc['status']!='NEW')&(df_changes_loc['created']>=min_date)]
    df_changes_loc = df_changes_loc.drop_duplicates(subset=['change_id'], keep='last')


    df_deps_red = df_dependencies_loc[['when_identified']]

    # Calculate Z-scores
    z_scores = np.abs((df_deps_red - df_deps_red.mean()) / df_deps_red.std())

    # Set a threshold for identifying outliers
    threshold = 3

    # Filter out the outliers
    df_clean = df_deps_red[(z_scores < threshold).all(axis=1)]

    df_dependencies_loc = df_dependencies_loc[df_dependencies_loc.index.isin(df_clean.index)]

    dependent_changes = set(df_dependencies_loc['Source'].tolist() + df_dependencies_loc['Target'].tolist())
    df_changes_loc = df_changes_loc[df_changes_loc['number'].isin(dependent_changes)]

    target_numbers_loc = df_changes_loc['number'].unique()

    global df_changes
    df_changes = df_changes_loc

    global target_numbers
    target_numbers = target_numbers_loc

    global df_dependencies
    df_dependencies = df_dependencies_loc


if __name__ == '__main__':
    
    print(f"Script {__file__} started...")
    
    start_date, start_header = hpr.generate_date("This script started at")

    init_global_vars()

    tscv = TimeSeriesSplit(n_splits = 10)
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_folds, fold, train_idx, test_idx) for fold, (train_idx, test_idx) in enumerate(tscv.split(target_numbers))]

        for out in concurrent.futures.as_completed(results):
            print(out.result())
    
    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print(f"Script {__file__} ended\n")
