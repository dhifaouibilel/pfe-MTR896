import os
import os.path as osp
from datetime import timedelta
import pandas as pd
import numpy as np
import concurrent.futures
import ast
import re
import utils.helpers as hpr
from utils import constants


DESCRIPTION_METRICS = constants.DESCRIPTION_METRICS

df_changes = None
df_deps = None
dependent_changes = None
cross_pro_changes = None
within_pro_changes = None

def count_project_age(row):
    project_creation_date = df_changes.loc[df_changes['project'] == row['project'], 'created'].iloc[0]
    change_creation_date = row['created']
    return (change_creation_date - project_creation_date).total_seconds()


def count_project_changes(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['created'] < row['created']), 
        'number'
    ].nunique()

def count_whole_changes(row):
    return df_changes.loc[
        df_changes['created'] < row['created'], 
        'number'
    ].nunique()


def count_project_changes_owner(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_whole_changes_owner(row):
    return df_changes.loc[
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_projects_contributed(row):
    return df_changes.loc[
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'project'
    ].nunique()


def count_projects_changes_deps(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['is_dependent'] == True) &
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_cross_project_changes(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['is_cross'] == True) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_within_project_changes(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['is_dependent'] == True) &
        (df_changes['is_cross'] == False) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_whole_cross_project_changes(row):
    return df_changes.loc[
        (df_changes['is_cross'] == True) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_whole_within_project_changes(row):
    return df_changes.loc[
        (df_changes['is_dependent'] == True) &
        (df_changes['is_cross'] == False) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_cross_pro_changes_owner(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['is_cross'] == True) &
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


def count_within_pro_changes_owner(row):
    return df_changes.loc[
        (df_changes['project'] == row['project']) &
        (df_changes['is_dependent'] == True) &
        (df_changes['is_cross'] == False) &
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()

# def count_hist_dep_pro(row):
#     projects = set(hpr.flatten_list(df_deps.loc[
#         (
#             ((df_deps['Source_repo'] == row['project'])&
#             (df_deps['Source_date'] < row['created'])) |
#             (df_deps['Target_repo'] == row['project'])&
#             (df_deps['Target_date'] < row['created'])
#          ) &
#         (df_deps['is_cross']==True),
#         ['Target_repo', 'Source_repo']
#     ].values))

#     return len(projects) - 1 if len(projects) != 0 else 0

def count_last_x_days_dependent_projects(row, days):
    days_ago = row['created'] - timedelta(days=days)
    projects = set(hpr.flatten_list(df_deps.loc[
        (
            ((df_deps['Source_repo'] == row['project'])&
            (df_deps['Source_date'] >= days_ago)&
            (df_deps['Source_date'] < row['created'])) |
            (df_deps['Target_repo'] == row['project'])&
            (df_deps['Target_date'] >= days_ago)&
            (df_deps['Target_date'] < row['created'])
         ) &
        (df_deps['is_cross']=='Cross'),
        ['Target_repo', 'Source_repo']
    ].values))

    return len(projects) - 1 if len(projects) != 0 else 0

def count_avg_cro_proj_nbr(row):
    df_deps_sub = df_deps.loc[
        (df_deps['is_cross'] == 'Cross') &
        (
            ((df_deps['Source_repo'] == row['project']) & 
             (df_deps['Source_date'] < row['created'])
            ) |
            ((df_deps['Target_repo'] == row['project']) & 
             (df_deps['Target_date'] < row['created']))
        ),
        ['Target_repo', 'Source_repo', 'is_cross']
    ]

    df_deps_sub['project'] = df_deps_sub.apply(lambda x: x['Target_repo'] if row['project'] == x['Source_repo'] else x['Source_repo'], axis=1)

    # Group by project and count the number of cross-project changes
    cross_project_changes_per_project = df_deps_sub.groupby('project')['is_cross'].sum()

    # Calculate the average number of cross-project changes across all projects
    average_cross_project_changes = cross_project_changes_per_project.mean()

    return average_cross_project_changes


def count_yearly_cro_proj_nbr(row):
    df_sub = df_changes.loc[
        (df_changes['created'] < row['created']) &
        (df_changes['project'] < row['project']) &
        (df_changes['is_cross'] == True)
        ,
        ['Source', 'Target', 'Target_repo', 'Source_repo', 'is_cross']
    ]

    df_sub['year'] = df_sub.apply(lambda x: x['created'].dt.year)

    # Group by year and project and count the number of cross-project changes
    cross_project_changes_per_year_project = df_sub.groupby(['year'])['is_cross'].sum()

    # Calculate the average number of cross-project changes across all projects and years
    average_cross_project_changes = cross_project_changes_per_year_project.mean(level='year')


    return average_cross_project_changes


# def count_last_thre_cro_proj_nbr(row):
#     # Filter changes within the last three months
#     three_months_ago = row['created'] - timedelta(days=3*30) # Assuming a month has 30 days
#     df_deps_sub = df_deps.loc[
#         (df_deps['is_cross'] == True) &
#         ((
#             (df_deps['Source'] == row['number']) & 
#             (df_deps['Source_date'] < row['created']) &
#             (df_deps['Source_date'] >= three_months_ago)
#         ) |
#         (
#             (df_deps['Target'] == row['number']) & 
#             (df_deps['Target_date'] < row['created']) &
#             (df_deps['Target_date'] >= three_months_ago)
#         )),
#         ['Source', 'Target', 'Target_repo', 'Source_repo', 'is_cross']
#     ]

#     df_deps_sub['number'] = df_deps_sub.apply(lambda x: x['Source'] if row['project'] == x['Source_repo'] else x['Target'])

#     return df_deps_sub['number'].nunique()


def count_last_x_days_cross_project_changes(row, days):
    days_ago = row['created'] - timedelta(days=days)
    unique_projects_nbr = df_changes.loc[
            (df_changes['project'] == row['project'])&
            (df_changes['is_cross'] == True)&
            (df_changes['created'] >= days_ago)&
            (df_changes['created'] < row['created']),
        'number'
    ].nunique()

    return unique_projects_nbr


def count_last_x_days_modified_unique_projects(row, days):
    days_ago = row['created'] - timedelta(days=days)
    unique_projects_nbr = df_changes.loc[
            (df_changes['created'] >= days_ago)&
            (df_changes['created'] < row['created']),
        'project'
    ].nunique()

    return unique_projects_nbr


def count_whole_changes_deps(row):
    return df_changes.loc[
        (df_changes['number'].isin(dependent_changes)) &
        (df_changes['owner_account_id'] == row['owner_account_id']) &
        (df_changes['created'] < row['created']),
        'number'
    ].nunique()


# def load_dependent_changes(project_type=None):
#     if project_type == 'Cross':
#         result = set(hpr.flatten_list(df_deps.loc[df_deps['is_cross']==True, ['Source', 'Target']].values))
#         return result
#     elif project_type == 'Within':
#         cross_pro_changes = set(hpr.flatten_list(df_deps.loc[df_deps['is_cross']==True, ['Source', 'Target']].values))
#         dep_changes = set(hpr.flatten_list(df_deps[['Source', 'Target']].values))
#         return dep_changes.difference(cross_pro_changes)
#     else:
#         return set(hpr.flatten_list(df_deps[['Source', 'Target']].values))


def count_num_file_types(changed_files):
    visited_extensions = []

    # Iterate through the files in the directory.
    for filename in changed_files:
        # Extract the file extension.
        file_extension = os.path.splitext(filename)[1].lower()

        # Increment the count for the file extension.
        if file_extension not in visited_extensions:
            visited_extensions.append(file_extension)
    return len(visited_extensions)


def count_num_directory_files(changed_files):
    visited_dirs = []

    # Iterate through the files in the directory.
    for filename in changed_files:
        # Extract the file extension.
        file_dir = os.path.splitext(filename)[0].lower()

        # Increment the count for the file dir.
        if file_dir not in visited_dirs:
            visited_dirs.append(file_dir)
    return len(visited_dirs)

# def count_num_programming_languages(changed_files):
#     # Create a set to store detected programming languages.
#     detected_languages = set()

#     # Iterate through the files in the directory.
#     for filename in changed_files:
#         # Read the file content.
#         # with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
#         #     content = file.read()

#         # Use pygments to guess the lexer (programming language).
#         try:
#             lexer = guess_lexer(filename)
#             detected_languages.add(lexer.name)
#         except:
#             # Handle exceptions if the lexer cannot be determined.
#             pass
#     return detected_languages


def count_desc_length(desc):
    return len(desc)


def count_words(text):
    return len(re.sub(r'\s+', ' ', text).split())


def identify_desc_nature(desc, keyword):
    words = re.sub(r'\s+', ' ', desc.lower()).split()
    change_type = DESCRIPTION_METRICS[keyword]
    result = any(word for word in words
                 if word in change_type['inclusion'] or
                 (change_type['pattern'] and re.search(
                     change_type['pattern'], word))
                 )
    return int(result)


def count_num_dev_modified_files(row):
    res = df_changes.loc[(df_changes['project']==row['project'])&(df_changes['created']<=row['created']), ["changed_files", "owner_account_id"]]
    res = res.explode('changed_files')
    res = res[res['changed_files'].isin(row['changed_files'])]
    return res['owner_account_id'].nunique()


def count_avg_num_dev_modified_files(row):
    res = df_changes[(df_changes['project']==row['project'])&(df_changes['created']<=row['created'])]
    res = res.explode('changed_files')
    res = res[res['changed_files'].isin(row['changed_files'])]
    num_devs = res['owner_account_id'].nunique()
    return num_devs / len(row['changed_files']) if len(row['changed_files']) > 0 else 0

def count_ratio_dep_chan_owner(row):
    owner_pro_cha = df_changes.loc[
       (df_changes['project']==row['project'])&
       (df_changes['owner_account_id']==row['owner_account_id'])&
       (df_changes['is_dependent']==True)&
       (df_changes['created']<=row['created']),
       "number"
       ].nunique()
   
    owner_all_cha = df_changes.loc[
       (df_changes['owner_account_id']==row['owner_account_id'])&
       (df_changes['is_dependent']==True)&
       (df_changes['created']<=row['created']),
       "number"
       ].nunique()
    
    return 100 * (owner_pro_cha/owner_all_cha) if owner_all_cha != 0 else 0

def count_num_modified_file_dependent_changes(row, attr):
    changed_files = row['changed_files']
    sub_operations = {"min": min, "max": max, "median": np.median, "mean": np.mean}
    if len(changed_files) == 0:
        for label in sub_operations.keys():
            row[f"{label}_{attr}"] = 0
        row[attr] = 0
        return row
    changed_files = {f: 0 for f in changed_files}
    for f in changed_files.keys():
        changed_files[f] = df_changes.loc[
            (df_changes['is_dependent']==True) &
            (df_changes['project']==row['project']) &
            (df_changes['created']<row['created']) &
            (df_changes['changed_files'].apply(lambda x: f in x)), 
        "number"].nunique()

    for label, func in sub_operations.items():
        row[f"{label}_{attr}"] = func(list(changed_files.values()))
    row[attr] = len([count for _, count in changed_files.items() if count > 0])
    pd.DataFrame({k: [v] for k,v in row.items()}).to_csv(osp.join('.', 'Files', 'Metrics', attr, f"{row['number']}.csv"))
    return row['number']

def count_num_file_changes(row):
    res = df_changes[(df_changes['project']==row['project'])&(df_changes['created']<=row['created'])]
    res = res.explode('changed_files')
    return res.loc[res['changed_files'].isin(row['changed_files']), 'number'].nunique()


def count_num_build(row, attr):
    messages = row['messages']
    if type(messages) != list:
        return 0
    num_build = 0
    for msg in messages:
        if attr in msg['message']:
            num_build += 1
     
    return num_build

def count_num_recent_branch_files_changes(row):
    res = df_changes[(df_changes['project']==row['project'])&(df_changes['created']<=row['created'])]
    res = res.explode('changed_files')
    return res.loc[res['changed_files'].isin(row['changed_files']), 'branch'].nunique()

def count_type_changes(row, type):
    return df_changes.loc[
        (df_changes['status']==type)&
        (df_changes['project']==row['project'])&
        (df_changes['created']<row['created']),
        'number'
    ].nunique()

def get_path(metric):
    return osp.join('.', 'Files', f'{metric}.csv')


def process_metrics(attr):
    global df_changes
    print(f'Processing {attr} metric started...')
    main_dev_attr = ['project', 'created', 'changed_files', 'owner_account_id', 'status']
    if attr == 'project_age':
        df_changes[attr] = df_changes.apply(count_project_age, axis=1)
    elif attr == 'project_changes_owner':
        df_changes[attr] = df_changes.apply(count_project_changes_owner, axis=1)
    elif attr == 'whole_changes_owner':
        df_changes[attr] = df_changes.apply(count_whole_changes_owner, axis=1)
    elif attr == 'projects_contributed':
        df_changes[attr] = df_changes.apply(count_projects_contributed, axis=1)
    elif attr == 'projects_changes_deps':
        df_changes[attr] = df_changes.apply(count_projects_changes_deps, axis=1)
    elif attr == 'whole_changes_deps':
        df_changes[attr] = df_changes.apply(count_whole_changes_deps, axis=1)
    elif attr == 'project_changes_count':
        df_changes[attr] = df_changes.apply(count_project_changes, axis=1)
    elif attr == 'whole_changes_count':
        df_changes[attr] = df_changes.apply(count_whole_changes, axis=1)
    elif attr == 'cross_project_changes':
        df_changes[attr] = df_changes.apply(count_cross_project_changes, axis=1)
    elif attr == 'within_project_changes':
        df_changes[attr] = df_changes.apply(count_within_project_changes, axis=1)
    # elif attr == 'whole_cross_project_changes':
    #     df_changes[attr] = df_changes.apply(count_whole_cross_project_changes, axis=1)
    elif attr == 'whole_within_project_changes':
        df_changes[attr] = df_changes.apply(count_whole_within_project_changes, axis=1)
    elif attr == 'cross_project_changes_owner':
        df_changes[attr] = df_changes.apply(count_cross_pro_changes_owner, axis=1)
    elif attr == 'within_project_changes_owner':
        df_changes[attr] = df_changes.apply(count_within_pro_changes_owner, axis=1)
    elif attr == 'last_mth_dep_proj_nbr':
        df_changes[attr] = df_changes.apply(count_last_x_days_dependent_projects, args=(30,), axis=1)
    elif attr == 'avg_cro_proj_nbr':
        df_changes[attr] = df_changes.apply(count_avg_cro_proj_nbr, axis=1)
    elif attr == 'last_mth_cro_proj_nbr':
        df_changes[attr] = df_changes.apply(count_last_x_days_cross_project_changes, args=(30,), axis=1)
    elif attr == 'last_mth_mod_uniq_proj_nbr':
        df_changes[attr] = df_changes.apply(count_last_x_days_modified_unique_projects, args=(30,), axis=1)

    # Change dimension
    elif attr == 'code_churn': 
        df_changes['code_churn'] = df_changes['insertions'] + df_changes['deletions']
    elif attr == 'num_file_types':
        df_changes['num_file_types'] = df_changes['changed_files'].map(count_num_file_types)
    elif attr == 'num_directory_files':
        df_changes['num_directory_files'] = df_changes['changed_files'].map(count_num_directory_files)

    # Text dimension
    elif attr == 'subject_length':
        df_changes[attr] = df_changes['subject'].map(len)
    elif attr == 'description_length':
        df_changes[attr] = df_changes['commit_message'].map(count_desc_length)
    elif attr == 'subject_word_count':
        df_changes[attr] = df_changes['subject'].map(count_words)
    elif attr == 'description_word_count':
        df_changes[attr] = df_changes['commit_message'].map(count_words)
    elif attr in DESCRIPTION_METRICS.keys():
        df_changes[attr] = df_changes['commit_message'].apply(lambda msg: identify_desc_nature(msg, attr))
    
    # File dimension
    elif attr == 'num_dev_modified_files':
        df_changes[attr] = df_changes[main_dev_attr].apply(count_num_dev_modified_files, axis=1)
    elif attr == 'avg_num_dev_modified_files':
        df_changes[attr] = df_changes[main_dev_attr].apply(count_avg_num_dev_modified_files, axis=1)
    elif attr == 'num_file_changes':
        df_changes[attr] = df_changes[main_dev_attr + ['number']].apply(count_num_file_changes, axis=1)
    elif attr == 'num_build_failures':
        df_changes[attr] = df_changes[main_dev_attr + ['number']].apply(count_num_build, args=('Build failed',), axis=1)
    elif attr == 'num_build_successes':
        df_changes[attr] = df_changes[main_dev_attr + ['number']].apply(count_num_build, args=('Build succeeded',), axis=1)
    elif attr == 'num_merged_changes':
        df_changes[attr] = df_changes[main_dev_attr + ['number']].apply(count_type_changes, args=('MERGED',), axis=1)
    elif attr == 'num_abandoned_changes':
        df_changes[attr] = df_changes[main_dev_attr + ['number']].apply(count_type_changes, args=('ABANDONED',), axis=1)
    elif attr == 'ratio_dep_chan_owner':
        df_changes[attr] = df_changes[main_dev_attr + ['number']].apply(count_ratio_dep_chan_owner, axis=1)
    elif attr == 'num_mod_file_dep_cha':
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(count_num_modified_file_dependent_changes, row, attr) for _, row in df_changes[main_dev_attr + ['number']].iterrows()]

            for f in concurrent.futures.as_completed(results):
                attr = f.result()
                print(f'{attr} ID processed successfully...')
    
    compact_columns = ['num_mod_file_dep_cha']
    if attr in compact_columns:
        columns = [f"{sub_col}_{attr}" for sub_col in ['min', 'max', 'mean', 'median']] + ['number', attr]
    else:
        columns = ["number"] + [attr]

    df_changes[columns].to_csv(
        osp.join('.', 'Files', 'Metrics', f'{attr}.csv'), index=None)
    
    print(f'{attr}.csv file saved successfully...')

    return attr


def init_global_vars():
    df = hpr.combine_openstack_data()
    df = df[['number', 'project', 'created', 'owner_account_id', 'branch', 'changed_files', 'insertions', 'deletions', 'subject', 'commit_message', 'status']]
    df['changed_files'] = df['changed_files'].map(ast.literal_eval)
    # df['messages'] = df.loc[df['messages'].notnull()==True, 'messages'].map(ast.literal_eval)

    df_dep_changes = pd.read_csv(osp.join('.', 'Files', 'all_dependencies.csv'))
    df_dep_changes = df_dep_changes[(df_dep_changes['Source_status']!='NEW')&(df_dep_changes['Target_status']!='NEW')]
    df_dep_changes['Source_date'] = pd.to_datetime(df_dep_changes['Source_date'])
    df_dep_changes['Target_date'] = pd.to_datetime(df_dep_changes['Target_date'])

    dependent_changes_loc = set(hpr.flatten_list(df_dep_changes[['Source', 'Target']].values))

    cross_pro_changes_loc = set(hpr.flatten_list(df_dep_changes.loc[df_dep_changes['is_cross']=='Cross', ['Source', 'Target']].values))
    within_pro_changes_loc = dependent_changes_loc.difference(cross_pro_changes_loc)

    df['is_dependent'] = df['number'].map(lambda nbr: True if nbr in dependent_changes_loc else False)
    df['is_cross'] = df['number'].map(lambda nbr: True if nbr in cross_pro_changes_loc else False)

    global df_changes
    df_changes = df
    
    global df_deps
    df_deps = df_dep_changes

    global dependent_changes
    dependent_changes = dependent_changes_loc

    global cross_pro_changes
    cross_pro_changes = cross_pro_changes_loc

    global within_pro_changes
    within_pro_changes = within_pro_changes_loc


if __name__ == '__main__':
    print(f"Script {__file__} started...")

    start_date, start_header = hpr.generate_date("This script started at")

    init_global_vars()

    metrics = constants.CHANGE_METRICS + constants.TEXT_METRICS +  + constants.PROJECT_METRICS 
    + constants.FILE_METRICS + constants.DEVELOPER_METRICS
    process_metrics(metrics[0])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_metrics, m) for m in metrics]

        for f in concurrent.futures.as_completed(results):
            attr = f.result()
            print(f'{attr} metric processed successfully...')

    end_date, end_header = hpr.generate_date("This script ended at")

    print(start_header)

    print(end_header)

    hpr.diff_dates(start_date, end_date)

    print(f"Script {__file__} ended\n")
