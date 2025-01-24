import os
import os.path as osp
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

from utils import helpers as hpr


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
    metric_list = [f for f in hpr.list_file(metric_path)]
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
    df['pctg_cross_project_changes_owner'] = df.apply(compute_ptg_cross_project_changes_owner, axis=1)
    
    return df

def is_cross_project(number, cross_pro_changes, within_pro_changes):
    if number in cross_pro_changes:
        return 1
    elif number in within_pro_changes:
        return 0
    else:
        return 2


def load_classifiers():
    # Base estimators
    rf_clf = RandomForestClassifier(random_state=42)
    et_clf = ExtraTreeClassifier(random_state=42)

    rnd_clf = RandomForestClassifier(random_state=42)
    ada_clf = AdaBoostClassifier(random_state=42)
    xgb_clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    estimators = [('rf', rf_clf), ('xgb', xgb_clf), ('et', et_clf)]

    max_clf = VotingClassifier(
        estimators=estimators,
        voting='hard')

    avg_clf = VotingClassifier(
        estimators=estimators,
        voting='soft')

    mlp_clf = MLPClassifier(random_state=42)

    ensemble_classifiers = {
        'ET': et_clf,
        'RF': rnd_clf,
        'XGBoost': xgb_clf,
        'AdaBoost': ada_clf,
        # 'AV': avg_clf,
        # 'MV': max_clf,
        'MLP': mlp_clf
    }

    return ensemble_classifiers
    

def correlation_analysis(X, metric_imp):
    # Calculate Spearman correlation matrix
    correlation_matrix, _ = spearmanr(X)

    highly_correlated_features = []
    columns = X.columns
    # Iterate through each pair of features
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            # If correlation is below 0.7, add both features
            corr = abs(correlation_matrix[i, j])
            if corr >= 0.7:
                # If correlation is above 0.7, select the feature with less importance score
                # print(columns[i], columns[j])
                col_i = columns[i]
                col_j = columns[j]
                # importance_i = metric_imp[col_i]
                # importance_j = metric_imp[col_j]
                # selected_feature = col_i if importance_i < importance_j else col_j
                # if selected_feature not in highly_correlated_features:
                    # highly_correlated_features.append(selected_feature)
                highly_correlated_features.append(f"{col_i}, {col_j}")

    return highly_correlated_features


def calculate_r_squared(X, y):
    """
    Calculate R-squared value for a linear regression model.
    
    Parameters:
    X : array-like
        The independent variables.
    y : array-like
        The dependent variable.

    Returns:
    r_squared : float
        R-squared value of the linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    r_squared = model.score(X, y)
    return r_squared

def redundancy_analysis(data):
    """
    Compute R-squared values for each feature predicting the remaining ones.
    
    Parameters:
    data : DataFrame
        The dataset containing all variables.

    Returns:
    r_squared_values : DataFrame
        DataFrame containing R-squared values for each feature predicting the remaining ones.
    """
    r_squared_values = {}

    for feature in data.columns:
        X = data.drop(columns=[feature])
        y = data[feature]
        r_squared = calculate_r_squared(X, y)
        r_squared_values[feature] = r_squared

    r_squared_df = pd.DataFrame(list(r_squared_values.items()), columns=['Feature', 'R_squared'])
    r_squared_df = r_squared_df.loc[r_squared_df['R_squared'] >= 0.9, 'Feature'].values.tolist()
    return r_squared_df


def doc2vec_model(df_changes, changes, fold, attr="commit_message"):
    """Doc2Vec model for training
    """
    model_path = osp.join(os.getcwd(), 'doc2vec', f'{fold}_{attr}')

    if osp.isfile(model_path) == True:
        return Doc2Vec.load(model_path)

    df_sub = df_changes.loc[
        df_changes['number'].isin(hpr.flatten_list(changes))&
        df_changes[attr].notnull(), 
        ['number', attr]
    ]

    print(f'Building Doc2Vec model for {model_path}...')
    # Example training data
    documents = df_sub.set_index('number')[attr].to_dict()

    # Preprocess documents and create tagged data
    tagged_data = [TaggedDocument(words=word_tokenize(value), tags=[str(key)]) for key, value in documents.items()]

    # Train the Doc2Vec model
    doc_model = Doc2Vec(vector_size=128, min_count=1, epochs=20)
    doc_model.build_vocab(tagged_data)
    doc_model.train(tagged_data, total_examples=doc_model.corpus_count, epochs=doc_model.epochs)
    
    doc_model.save(model_path)

    print(f'Doc2Vec model for {model_path} fold built successfully')

    return doc_model

def calc_cosine_similarity(embed):
    A = embed[0]
    B = embed[1]

    if (type(A) == float) or (type(B) == float) or (len(A) == 0) or (len(B) == 0):
        return 0 
    cosine = np.dot(A,B) / (norm(A) * norm(B))
 
    return cosine

def to_embedding(x, model):
    if type(x) == float:
        return []
    return model.infer_vector(word_tokenize(x))


def compute_embdedding_similarity(df_changes, model, X, attr, label):
    train_test_changes = df_changes.loc[
        df_changes['number'].isin(hpr.flatten_list(X[['Source', 'Target']].values)),
        ['number', attr]
    ]

    ### Compute embedding of change's description
    # print(train_test_changes.iloc[0, :])
    train_test_changes[f'{label}_embed'] = train_test_changes[attr].apply(to_embedding, args=(model,))
    # train_test_changes['add_lines_embed'] = train_test_changes['added_lines'].map(lambda x: model.infer_vector(word_tokenize(x)))
    # train_test_changes['del_lines_embed'] = train_test_changes['deleted_lines'].map(lambda x: model.infer_vector(word_tokenize(x)))

    X = pd.merge(
        left=X, 
        right=train_test_changes, 
        left_on='Source', 
        right_on='number', 
        how='left',
        suffixes=('_target', '_source')
    )
    X = pd.merge(
        left=X, 
        right=train_test_changes, 
        left_on='Target', 
        right_on='number', 
        how='left',
        suffixes=('_target', '_source')
    )
    
    # Compute cosine similarity between the embeddings of source and target changes
    X[f'{label}_sim'] = X[[f'{label}_embed_source', f'{label}_embed_target']].apply(calc_cosine_similarity, axis=1)
    # X['add_lines_sim'] = X[['add_lines_embed_source', 'add_lines_embed_target']].apply(calc_cosine_similarity, axis=1)
    # X['del_lines_sim'] = X[['del_lines_embed_source', 'del_lines_embed_target']].apply(calc_cosine_similarity, axis=1)

    # X['shared_tokens'] = X[['changed_files_source', 'changed_files_target']].apply(compute_shared_tokens, axis=1)
   
    X.drop(columns=[
            f'{label}_embed_source', 
            f'{label}_embed_target', 
            f'{attr}_source', 
            f'{attr}_target',
            'number_target', 
            'number_source'
        ], 
    inplace=True)

    # X['desc_sim'].to_csv('test.csv', index=None)
    # pd.DataFrame({'col': X.columns.tolist()}).to_csv('test.csv', index=None)

    return X


def compute_filenames_shared_tokens(row, changed_files):
    file_names_src = set(changed_files[row['Source']])
    file_names_trg = set(changed_files[row['Target']])
    intersect = len(file_names_src.intersection(file_names_trg))
    union = len(file_names_src.union(file_names_trg))
    return intersect/union if union != 0 else 0


def compute_shared_desc_tokens(row, changes_description):
    desc_src = changes_description[row['Source']]
    desc_trg = changes_description[row['Target']]

    if not desc_src or not desc_trg:
        return 0
    
    desc_src = set(word_tokenize(changes_description[row['Source']]))
    desc_trg = set(word_tokenize(changes_description[row['Target']]))
    intersect = len(desc_src.intersection(desc_trg))
    union = len(desc_src.union(desc_trg))
    return intersect/union if union != 0 else 0