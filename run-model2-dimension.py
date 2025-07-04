import os
import os.path as osp
import argparse
import numpy as np
import pandas as pd
import concurrent.futures
from xgboost import XGBClassifier
from sklearn.metrics import  precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, brier_score_loss
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from utils import constants
from utils import helpers as hpr
import utils.classifier_util as clas_util
from logging_config import get_logger

# df: pd.DataFrame = None
df_features: pd.DataFrame = None
dimensions: dict = None

logger = get_logger(log_filename='dimension-perf')

def process_dimension(dim_label: str, dim_feats: list, dimension_type: str='keep'):
    logger.info(f'Start training with {dim_label} dimension...')
    
    features = []
    if dimension_type == 'keep':
        features = dim_feats
    else:
        for lab, dim in dimensions.items():
            if lab != dim_label:
                features += dim

    auc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    f2_scores = []
    brier_scores = []
    corr_features = []

    for fold in range(0, 10):
        clone_clf = XGBClassifier(random_state=42)
        
        # Split training data into features and dims
        X_train = pd.read_csv(osp.join(".", "Files", "Data", "Train", f"{fold}.csv"))
        y_train = X_train['related']

        corr_features = df_features.loc[df_features[f'Fold{fold}']==0, 'Feat'].tolist()
        features = [c for c in features if c not in corr_features]

        X_train = X_train[features]

        X_test = pd.read_csv(osp.join(".", "Files", "Data", "Test", f"temp_{fold}.csv"))
        # X_test_pairs = X_test[['Source', 'Target', 'related']]
        y_test = X_test['related']

        X_test = X_test[X_train.columns.tolist()]

        # Train the Random Forest Classifier on the training fold set 
        clone_clf.fit(X_train, y_train)

        # Test the Random Forest Classifier on the test fold set 
        y_probs = clone_clf.predict_proba(X_test)[:,1]

        # Set custom threshold
        threshold = 0.75
        y_pred = [1 if p >= threshold else 0 for p in y_probs]

        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        f2_scores.append(fbeta_score(y_test, y_pred, beta=2))
        auc_scores.append(roc_auc_score(y_test, y_pred))
        brier_scores.append(brier_score_loss(y_test, y_pred))

        logger.info(f"{dim_label}, Fold: {fold+1}, Precision: {precision_scores[-1]}, Recall: {recall_scores[-1]}, F1: {f1_scores[-1]}, F2: {f2_scores[-1]}, AUC: {auc_scores[-1]}, Brier: {brier_scores[-1]}")

    # feature_importances /= (fold+1)
    prec_avg = np.average(precision_scores)
    recall_avg = np.average(recall_scores)
    f1_avg = np.average(f1_scores)
    f2_avg = np.average(f2_scores)
    auc_avg = np.average(auc_scores)
    brier_avg = np.average(brier_scores)

    logger.info(f"{dim_label}, Precision: {prec_avg}, Recall: {recall_avg}, F1: {f1_avg}, F2: {f2_avg}, AUC: {auc_avg}, Brier: {brier_avg}")

    return {
        'Dimension': dim_label,
        'Precision': prec_avg,
        'Recall': recall_avg,
        'F1': f1_avg,
        'F2': f2_avg,
        'AUC': auc_avg,
        'Brier': brier_avg
    }

def init_global_vars():
    df = clas_util.combine_features()
    df = df.drop(columns=['number', 'num_build_failures'], errors='ignore')

    df_features_loc = pd.read_csv(osp.join(".", "Results", "Correlation", "second_model.csv"))

    M1_METRICS = df.columns.tolist()
    CHANGE_METRICS = [col for col in constants.CHANGE_METRICS if col in M1_METRICS]
    TEXT_METRICS = [col for col in constants.TEXT_METRICS if col in M1_METRICS]
    DEVELOPER_METRICS = [col for col in constants.DEVELOPER_METRICS if col in M1_METRICS]
    PROJECT_METRICS = [col for col in constants.PROJECT_METRICS if col in M1_METRICS]
    FILE_METRICS = [col for col in constants.FILE_METRICS if col in M1_METRICS]
    CHANGE_METRICS = [f'{cm}_source' for cm in CHANGE_METRICS] + [f'{cm}_target' for cm in CHANGE_METRICS]
    TEXT_METRICS = [f'{cm}_source' for cm in TEXT_METRICS] + [f'{cm}_target' for cm in TEXT_METRICS]
    DEVELOPER_METRICS = [f'{cm}_source' for cm in DEVELOPER_METRICS] + [f'{cm}_target' for cm in DEVELOPER_METRICS]
    PROJECT_METRICS = [f'{cm}_source' for cm in PROJECT_METRICS] + [f'{cm}_target' for cm in PROJECT_METRICS]
    FILE_METRICS = [f'{cm}_source' for cm in FILE_METRICS] + [f'{cm}_target' for cm in FILE_METRICS]

    dimensions_loc = {
        'Change': CHANGE_METRICS,
        'Text': TEXT_METRICS,
        'Developer': DEVELOPER_METRICS,
        'Project': PROJECT_METRICS,
        'File': FILE_METRICS,
        'Pairs': constants.PAIR_METRICS
    }

    # global df
    # df = df_loc

    global df_features
    df_features = df_features_loc
    
    global dimensions
    dimensions = dimensions_loc

def main(dimension_type: str):

    clf_path = osp.join('.', 'Results')

    if not os.path.exists(clf_path):
        os.makedirs(clf_path)

    dimension_results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(process_dimension, dim_label, dim_feats, dimension_type) for dim_label, dim_feats in dimensions.items()]

        for out in concurrent.futures.as_completed(results):
            logger.info(out.result())
            dimension_results.append(out.result())
            df_dim_perf = pd.DataFrame(dimension_results)
            df_dim_perf.to_csv(osp.join('.', 'Results', 'Feature_importance', f'second_model_{dimension_type}_dim_new.csv'), index=None)


if __name__ == '__main__':
    
    logger.info(f"Script {__file__} started...")

    parser = argparse.ArgumentParser(
        description="To change later on.",
        epilog="For additional assistance, read out to Ali Arabat",
    )
    parser.add_argument(
        "-d", "--dimension-type", type=str, default='keep'
    )

    args = parser.parse_args()
    args = vars(args)

    dimension_type = args['dimension_type']
    
    init_global_vars()
    main(dimension_type)
    
    start_date, start_header = hpr.generate_date("This script started at")

    
    end_date, end_header = hpr.generate_date("This script ended at")

    logger.info(start_header)

    logger.info(end_header)

    hpr.diff_dates(start_date, end_date)

    logger.info(f"Script {__file__} ended\n")
