import os.path as osp

DESCRIPTION = {
    'is_corrective': {
        'inclusion': ['bug', 'wrong', 'problem', 'patch', 'fixed', 'fixes', 'fixing', 'closed', 'closes', 'closing', 'bugs', 'error', 'errors', 'defect', 'defects', 'except', 'resolves', 'resolved', 'crash', 'make sure'],
        'pattern': r'^(?:fix|close|bug|issue|fault|fail|crash|fail|solve).*'
    },
    'is_refactoring': {
        'inclusion': ['clean', 'better', 'bugs', 'cleanup', 'cohesion', 'comment', 'complexity', 'consistency', 'decouple', 'duplicate', 'fix', 'performance', 'readability', 'redundancy', 'regression', 'risks', 'simply', 'testing', 'uncomment', 'modernize', 'modif', 'modularize', 'optimize', 'refactor', 'remove', 'rewrite', 'restructure', 'reorganize', 'generalise', '->', '=>', 'move', 'deprecated', 'deprecate', 'code', 'readability', 'test', 'bad code', 'bad management', 'best practice', 'break out', 'extract class', 'loose coupling', 'not documented', 'open close', 'splitting logic', 'strategy pattern', 'single responsibility', 'ease of use', 'improving code quality', 'remove legacy code', 'replace hard coded', 'stress test results', 'single level of abstraction per function', 'allow easier integration with'],
        'pattern': None
    },
    'is_preventive': {
        'inclusion': ['junit', 'coverage', 'asset', 'unit test', 'UnitTest', 'cppunit'],
        'pattern': r'^(?:test|unit|prevent).*'
    },
    'is_non_functional': {
        'inclusion': ['docs', 'document', 'documentation', 'help'],
        'pattern': r'^(?:doc).*'
    },
    'has_feature_addition': {
        'inclusion': ['new', 'add', 'requirement', 'initial', 'create', 'implement', 'implementing', 'insert', 'inserted', 'insertions'],
        'pattern': None
    },
    'is_merge': {
        'inclusion': ['merge', 'combine', 'unite', 'join', 'consolidate', 'incorporate', 'fuse'],
        'pattern': r'^(?:merg|mix).*'
    }
}

DEVELOPER_METRICS = [
    # Developer
    'cross_project_changes_owner',
    'within_project_changes_owner',
    'whole_changes_owner',
    'pctg_cross_project_changes_owner',
    'projects_contributed_owner',
    'project_changes_owner',
    'ratio_dep_chan_owner'
]

CHANGE_METRICS = [
    # Change dimension
    # 'whole_cross_project_changes',
    # 'last_three_mth_cro_proj_nbr',
    # 'last_six_mth_cro_proj_nbr',
    # 'last_year_cro_proj_nbr',
    'insertions',
    'deletions',
    'code_churn',
    'num_directory_files',
    # 'projects_changes_deps',
    # 'whole_changes_deps',
    # 'project_changes_count',
    # 'whole_changes_count'
]+ list(DESCRIPTION.keys())

PROJECT_METRICS = [
    # PROJECT
    'project_age',
    'last_mth_dep_proj_nbr',
    'last_mth_cro_proj_nbr',
    'cross_project_changes',
    'within_project_changes',
    'pctg_cross_project_changes',
    'whole_within_project_changes',
]

FILE_METRICS = [
    # File dimension
    'num_file_changes',
    'num_file_types',
    'num_dev_modified_files',
    'avg_num_dev_modified_files',
    'pctg_mod_file_dep_cha',
    'min_mod_file_dep_cha',
    'median_mod_file_dep_cha',
    'max_mod_file_dep_cha'
]
TEXT_METRICS = [
    # Text dimension
    'description_word_count',
    'description_length',
    'subject_word_count',
    'subject_length'
]

PAIR_METRICS = [
    # metrics for the third model for the target change
    "pctg_inter_dep_cha",
    "src_trgt_co_changed_nbr",
    "dev_in_src_change_nbr",
    "rev_in_src_change_nbr",
    "num_shrd_desc_tkns",
    "num_shrd_file_tkns",
    "desc_sim",
    "subject_sim",
    "add_lines_sim",
    "del_lines_sim"
]


def get_metrics():
    metrics = CHANGE_METRICS + DEVELOPER_METRICS + PROJECT_METRICS + \
        FILE_METRICS + TEXT_METRICS + PAIR_METRICS 

    return metrics


CHANGE_WITH_DEPENDENCY = False
DURATION = 365 * 2  # In days
TRAINING_PATH = osp.join('.', 'Files', 'training')
