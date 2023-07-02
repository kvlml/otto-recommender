import argparse
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping
from sklearn.model_selection import GroupKFold
import time
import gc

from utils import get_group_for_lgb, lgb_mrr_wrapper

parser = argparse.ArgumentParser()
parser.add_argument('--candidates', type=str, default='../../data/07_model_output/final_train_carts_orders_w_features_new_split.parquet')
parser.add_argument('--binary_target', type=int, default=True)
parser.add_argument('--user_col', type=str, default='session')
parser.add_argument('--target_col', type=str, default='target')
parser.add_argument('--problem_type', type=str, default='ranking')


def downsample_negative_lgb_dataset(train: lgb.Dataset, keep: float = 0.08):
    np.random.seed(33)
    train.construct()
    label = train.get_label()
    data = train.get_data()
    features = train.get_feature_name()
    cols_models = [
        col for col in features if "_score" in col or "_rank" in col
    ]
    data = data[:,:len(cols_models)]

    negatives = np.where(label == 0)[0]
    # positives = np.where(label > 0)[0]
    positives = np.where(
        (np.isnan(data).all(axis=1) == False)
        & (label > 0)
    )[0]
    num_negatives = int(np.ceil(len(negatives) * keep))

    print('num_positives:', len(positives))
    np.random.shuffle(negatives)
    new_negatives = negatives[:num_negatives]
    print('num_negatives:', num_negatives)
    print('num_negatives to num_positives:', num_negatives / len(positives))
    index = np.concatenate([positives, new_negatives])
    index.sort()
    return train.subset(index)


def preproc(train: lgb.Dataset, test: lgb.Dataset, params: dict):
    return downsample_negative_lgb_dataset(train), test, params


if __name__ == '__main__':
    args = parser.parse_args()
    train = pd.read_parquet(args.candidates)
    columns = [
        col for col in train.columns if col not in ['session', 'item_id', 'target']#if "_score" in col or "_rank" in col
    ]
#     print(columns)

#     df_target_stats = train.groupby("session")[["target"]].sum().reset_index()
#     df_target_stats = df_target_stats[df_target_stats["target"] > 0]
#     train = train[train["session"].isin(set(df_target_stats["session"].values))].reset_index(drop=True)
#     del df_target_stats
#     gc.collect()

    X_train = train[columns].to_numpy(dtype=np.float32)
    train[args.target_col] = (train[args.target_col] > 0) * 1
    y_train = train[args.target_col].astype(np.int8) if args.binary_target else train['target'].astype(np.float32)

    print('getting groups')
    train_groups = get_group_for_lgb(train[args.user_col].values)

    del train
    gc.collect()

    print('preparing Dataset')
    lgb_train = lgb.Dataset(
        X_train, y_train,
        feature_name=columns,
        group=train_groups,
        free_raw_data=False,
    )

    del X_train, y_train, train_groups
    gc.collect()

    objective = "lambdarank" if args.problem_type == 'ranking' else 'binary'
    metric = "map"
    eval_at = 3
    params = {
        "objective": objective,
        "metric": metric,
        "eval_at": eval_at,
        "boosting_type": "gbdt",
        # "max_depth": 4,
        # "num_leaves": 200,
        "is_unbalance": False,
        "learning_rate": 0.10,
        'lambda_l1': 0.9822808040431535,
        'lambda_l2': 7.840380968160073,
        'feature_fraction': 0.95,
        'feature_fraction_bynode': 0.8,
        'bagging_fraction': 0.95, #0.6133731265749973,
        'bagging_freq': 1,
        'n_jobs': 24,
        'deterministic': True,
        'force_row_wise': True,
        'random_state': 43,
    }

    print('doing cv')

    t = time.time()
    cv_results = lgb.cv(
        params,
        lgb_train,
        num_boost_round=590,
        folds=GroupKFold(3),
        shuffle=False,
        stratified=False,
        #feval=lgb_mrr_wrapper,
        fpreproc=preproc,
        callbacks=[
            early_stopping(50),
            lgb.log_evaluation(10)
        ],
        return_cvbooster=True,
    )
    print('cv duration:', round((time.time() - t) / 60, 3), 'minutes')
    print('saving models')
    for i, booster in enumerate(cv_results['cvbooster'].boosters):
        booster.save_model(f'booster_fold{i}.lgb')

    mean_metric = cv_results[f'{metric}@{str(eval_at)}-mean'][-1]
    std_metric = cv_results[f'{metric}@{str(eval_at)}-stdv'][-1]

    metrics = {'metric': mean_metric, 'std': std_metric}
    print('metric_avg:', metrics['metric'])
    with open('cv_metrics.json', 'w') as f:
        json.dump(metrics, f)
