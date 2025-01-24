import os.path as osp
import pandas as pd
from datetime import timedelta
from utils import helpers as hpr
from utils import constants
from utils import classifier_util as clas_util
import concurrent.futures
import ast

METRICS = [m for m in constants.get_metrics()]

# highly_correlated = [
#     'projects_changes_deps', 'whole_changes_count', 'whole_within_project_changes', 
#     'last_year_cro_proj_nbr', 'last_six_mth_cro_proj_nbr', 'last_mth_cro_proj_nbr', 
#     'last_six_mth_mod_uniq_proj_nbr', 'last_three_mth_mod_uniq_proj_nbr', 
#     'num_dev_modified_files', 'pctg_common_dev'
# ]

# METRICS = [m for m in METRICS if m not in highly_correlated]


df_dependent_changes = None
dependent_changes = None
cross_pro_changes = None
within_pro_changes = None
df = None
changed_files = None
changes_description = None
added_lines = None
deleted_lines = None


def compute_pctg_cross_project_changes(row):
    dominator = row['cross_project_changes'] + row['within_project_changes']
    if dominator == 0:
        return 0
    return row['cross_project_changes'] / dominator

def compute_pctg_whole_cross_project_changes(row):
    dominator = row['whole_cross_project_changes'] + row['whole_within_project_changes']
    if dominator == 0:
        return 0
    return row['whole_cross_project_changes'] / dominator

def compute_ptg_cross_project_changes_owner(row):
    dominator = row['cross_project_changes_owner'] + row['within_project_changes_owner']
    if dominator == 0:
        return 0
    return row['cross_project_changes_owner'] / dominator

def combine_features():
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

    df['project_age'] /= (60 * 60 * 24)

    df['pctg_cross_project_changes'] = df.apply(compute_pctg_cross_project_changes, axis=1)
    df['pctg_whole_cross_project_changes'] = df.apply(compute_pctg_whole_cross_project_changes, axis=1)
    df['pctg_cross_project_changes_owner'] = df.apply(compute_ptg_cross_project_changes_owner, axis=1)
    
    return df

def is_cross_project(number):
    if number in cross_pro_changes:
        return 1
    elif number in within_pro_changes:
        return 0
    else:
        return 2
    

def compute_common_dev_pctg(row):
    dev_source = df.loc[
        (df['project'] == row['project_source']) &
        (df['created'] < row['Target_date']),
        'owner_account_id'
    ].unique()
    dev_target = df.loc[
        (df['project'] == row['project_target']) &
        (df['created'] < row['Target_date']),
        'owner_account_id'
    ].unique()

    union = len(set(dev_source).union(dev_target))
    intersect = len(set(dev_source).intersection(dev_target))
    return intersect/union if union != 0 else 0

def count_dev_in_src_change(row):
    changes_nbr = df.loc[
        (df['project'] == row['project_source']) &
        (df['created'] < row['Target_date']) &
        (df['owner_account_id'] == row['owner_account_id_target']),
        'number'
    ].nunique()

    return changes_nbr

def count_rev_in_src_change(row):
    account_id = row['owner_account_id_target']
    reviewers = df.loc[
        (df['project'] == row['project_source']) &
        (df['created'] < row['Target_date']) & 
        (df['owner_account_id'] != account_id), 'reviewers'].values
    rev_exists = [account_id in reviewers_list for reviewers_list in reviewers]
    return sum(rev_exists)

def count_src_trgt_co_changed(row):
    return len(df_dependent_changes[
        (df_dependent_changes['Source_repo'] == row['project_source']) &
        (df_dependent_changes['Target_repo'] == row['project_target']) &
        (df_dependent_changes['Target_date'] < row['Target_date'])
    ])

def count_changed_files_overlap(row):
    files1 = changed_files[row['Source']]
    file2 = changed_files[row['Target']]
    
    return len(files1 & file2) / len(files1 | file2) if len(files1 | file2) > 0 else 0


def assign_past_changes(row, X):
    thirty_days_ago = row['created'] - timedelta(days=15)
    source_changes = X.loc[
        (X['project'] != row['project']) &
        # (X['is_cross'] == True) &
        (X['created'] < row['created']) &
        (X['created'] >= thirty_days_ago),
        'Target'
    ].tail(30).tolist()
    # if len(source_changes) >= 60:
    #     source_changes = random.sample(source_changes, 60)
    
    source_changes += df_dependent_changes.loc[
        (df_dependent_changes['is_cross']==True)& 
        (df_dependent_changes['Target']==row['Target']), 
        'Source'].tolist()
    return set(source_changes)



def build_cross_project_pairs(label, target):
    print(f'******** Started building pairs for fold {target} ********')
    path = osp.join('.', 'Files', 'Data', label, target)
    X = pd.read_csv(path)
    # X = df.loc[df['number'].isin(target), ['number', 'created', 'project', 'owner_account_id']]
    # X = pd.merge(
    #     left=X, 
    #     right=df[['number', 'created', 'project', 'owner_account_id']], 
    #     left_on='Target', 
    #     right_on='number', 
    #     how='left',
    #     suffixes=('_source', '_target')
    # )
    # X.rename(columns={'number': 'Target'}, inplace=True)
    # X['Source'] = X.apply(assign_past_changes, args=(X,), axis=1)
    # X = X.explode(column='Source')
    # X.drop(columns=['number'], inplace=True)
    # X.dropna(subset=['Source'], inplace=True)
    # print(f'******** Source attached for {target} ********')
    X = pd.merge(
        left=X, 
        right=df[['number', 'created', 'project', 'owner_account_id', 'reviewers']], 
        left_on='Source', 
        right_on='number', 
        how='left',
        suffixes=('_source', '_target')
    )
    # X.sort_values(by=['Target_date', 'Source_date'], inplace=True)
    # X.reset_index(drop=True, inplace=True)

    # X = pd.merge(
    #     left=X, 
    #     right=df_dependent_changes[['Source', 'Target', 'related']], 
    #     left_on=['Source', 'Target'], 
    #     right_on=['Source', 'Target'], 
    #     how='left'
    # )

    X = pd.merge(
        left=X, 
        right=df[METRICS + ['number']], 
        left_on='Source', 
        right_on='number', 
        how='inner',
        suffixes=('_target', '_source')
    )
    X = pd.merge(
        left=X, 
        right=df[METRICS + ['number']], 
        left_on='Target', 
        right_on='number', 
        how='inner',
        suffixes=('_target', '_source')
    )
    # X = pd.merge(
    #     left=X, 
    #     right=df[METRICS + ['number']], 
    #     left_on='Source', 
    #     right_on='number', 
    #     how='left',
    #     suffixes=('_target', '_source')
    # )
    # X = pd.merge(
    #     left=X, 
    #     right=df[METRICS + ['number']], 
    #     left_on='Target', 
    #     right_on='number', 
    #     how='left',
    #     suffixes=('_target', '_source')
    # )

    # X = pd.merge(
    #     left=X, 
    #     right=df[['number', 'owner_account_id', 'project']], 
    #     left_on='Source', 
    #     right_on='number', 
    #     how='left',
    #     suffixes=('_target', '_source')
    # )
    # X = pd.merge(
    #     left=X, 
    #     right=df[['number', 'owner_account_id', 'project']], 
    #     left_on='Target', 
    #     right_on='number', 
    #     how='left',
    #     suffixes=('_target', '_source')
    # )
 
    # X['diff_dev'] = X['owner_account_id_source'] != X['owner_account_id_target']
    # print(f'** diff_dev for {target} generated **')

    X['changed_files_overlap'] = X.apply(count_changed_files_overlap, axis=1)
    print(f'** changed_files_overlap for {target} generated **')

    X['cmn_dev_pctg'] = X.apply(compute_common_dev_pctg, axis=1)
    print(f'** cmn_dev_pctg for {target} generated **')

    X['num_shrd_file_tkns'] = X[['Source', 'Target']].apply(clas_util.compute_filenames_shared_tokens, args=(changed_files,), axis=1)
    print(f'** num_shrd_file_tkns for {target} generated **')
    
    X['num_shrd_desc_tkns'] = X[['Source', 'Target']].apply(clas_util.compute_shared_desc_tokens, args=(changes_description,), axis=1)
    print(f'** num_shrd_desc_tkns for {target} generated **')
    
    X['dev_in_src_change_nbr'] = X.apply(count_dev_in_src_change, axis=1)
    print(f'** dev_in_src_change_nbr for {target} generated **')
    
    X['rev_in_src_change_nbr'] = X.apply(count_rev_in_src_change, axis=1)
    print(f'** rev_in_src_change_nbr for {target} generated **')
    
    X['src_trgt_co_changed_nbr'] = X.apply(count_src_trgt_co_changed, axis=1)
    print(f'** src_trgt_co_changed_nbr for {target} generated **')


    X.drop(columns=['number_source', 'number_target', 'reviewers', 'project_source', 'project_target', 'owner_account_id_source', 'owner_account_id_target'], axis=1, inplace=True)
    
    X.to_csv(path, index=None)

    return target


def initialize_global_vars():

    df_deps = pd.read_csv(osp.join('.', 'Files', 'all_dependencies.csv'))
    df_deps['Source_date'] = pd.to_datetime(df_deps['Source_date'])
    df_deps['Target_date'] = pd.to_datetime(df_deps['Target_date'])
    df_deps['related'] = True

    global df_dependent_changes
    df_dependent_changes = df_deps

    dependent_changes_loc = set(hpr.flatten_list(df_deps[['Source', 'Target']].values))
    cross_pro_changes_loc = set(hpr.flatten_list(df_deps.loc[df_deps['is_cross']==True, ['Source', 'Target']].values))
    within_pro_changes_loc = dependent_changes_loc.difference(cross_pro_changes_loc)

    global dependent_changes
    dependent_changes = dependent_changes_loc

    global cross_pro_changes
    cross_pro_changes = cross_pro_changes_loc

    global within_pro_changes
    within_pro_changes = within_pro_changes_loc

    df_changes = hpr.combine_openstack_data()
    df_changes['reviewers'] = df_changes['reviewers'].map(ast.literal_eval)
    df_changes['reviewers'] = df_changes['reviewers'].map(lambda x: [rev['_account_id'] for rev in x])
    df_changes['changed_files'] = df_changes['changed_files'].map(hpr.combine_changed_file_names)
    df_changes['commit_message'] = df_changes['commit_message'].map(hpr.preprocess_change_description)

    df_features = combine_features()
    df_features = pd.merge(
        left=df_features, 
        right=df_changes[['number', 'created', 'project', 'owner_account_id', 'reviewers']], 
        left_on='number', 
        right_on='number', 
        how='inner',
        suffixes=('_source', '_target')
    )
    df_features['is_dependent'] = df_features['number'].map(lambda nbr: 1 if nbr in dependent_changes_loc else 0)
    df_features['is_cross'] = df_features['number'].map(is_cross_project)

    changed_files_loc = dict(zip(df_changes['number'], df_changes['changed_files']))
    changes_description_loc = dict(zip(df_changes['number'], df_changes['commit_message']))
    added_lines_loc = dict(zip(df_changes['number'], df_changes['added_lines']))
    deleted_lines_loc = dict(zip(df_changes['number'], df_changes['deleted_lines']))

    global changed_files
    changed_files = changed_files_loc

    global changes_description
    changes_description = changes_description_loc

    global added_lines
    added_lines = added_lines_loc

    global deleted_lines
    deleted_lines = deleted_lines_loc
    
    del df_changes

    global df
    df = df_features



if __name__ == '__main__':
    
    print(f"Script {__file__} started...")
    
    start_date, start_header = hpr.generate_date("This script started at")

    initialize_global_vars()

    # df_cros_pro_changes = df.loc[df['is_cross']==True].sort_values(by='created')
    # df_cros_pro_changes = df_cros_pro_changes['number'].values

    label = "Train"
    fold_files = [f for f in hpr.list_file(osp.join('.', 'Files', 'Data', label))]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(build_cross_project_pairs, label, cpc) for cpc in fold_files]

        for out in concurrent.futures.as_completed(results):
            print(f'Features for target-based pair {out.result()} saved to memory succesfully')
    
    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print(f"Script {__file__} ended\n")
