import pandas as pd
import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from tqdm.notebook import tqdm
import os
import gc
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Iterable, Any, Optional
from beartype import beartype
import warnings
from pandarallel import pandarallel
from numba import jit, njit, prange

tqdm.pandas()

Int = Union[int, np.int64]
Float = Union[float, np.float64]
Session = Union[List[Int], np.ndarray]
Sessions = List[Session]

# pandarallel.initialize(nb_workers=16, progress_bar=True, use_memory_fs=False)

warnings.filterwarnings("ignore")


from functools import wraps
import time


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        total_time = (te - ts) / 60
        print('func %r took: %2.4f min' % \
          (f.__name__, total_time))
        return result
    return wrap


@jit(nopython=True)
def mrr(preds, targets, groups):
    total = 0
    group_start = 0

    for group_size in groups:
        ranks = np.argsort(preds[group_start:group_start + group_size])[::-1]
        for i in range(min(len(ranks), 100)):
            if targets[group_start + ranks[i]] == 1:
                total += 1 / (i + 1)

        group_start = group_start + group_size

    return total / len(groups)


@njit(parallel=True)
def pmrr(preds, targets, groups):
    total = 0
    group_starts = np.cumsum(groups)

    for group_id in prange(len(groups)):
        group_end = group_starts[group_id]
        group_start = group_end - groups[group_id]
        ranks = np.argsort(preds[group_start: group_end])[::-1]
        for i in range(min(len(ranks), 100)):
            if targets[group_start + ranks[i]] == 1:
                total += 1 / (i + 1)

    return total / len(groups)


def lgb_mrr_wrapper(preds, lgb_dataset):
    metric = pmrr(preds, lgb_dataset.label, lgb_dataset.group)
    return 'mrr', metric, True


@beartype
def downsample_negative(df: pd.DataFrame, keep: float, target_col: str) -> pd.DataFrame:
    np.random.seed(33)
    negatives = np.where((df[target_col] == 0).values)[0]
    positives = np.where((df[target_col] > 0).values)[0]
    num_negatives = int(np.ceil(len(negatives) * keep))
    np.random.shuffle(negatives)
    new_negatives = negatives[:num_negatives]
    index = np.concatenate([positives, new_negatives])
    index.sort()
    return df.iloc[index]


@beartype
def get_group_for_lgb(sorted_vals: Iterable) -> np.ndarray:
    curr_val = sorted_vals[0]
    cnt = 1
    groups = []
    for val in sorted_vals[1:]:
        if val != curr_val:
            groups.append(cnt)
            curr_val = val
            cnt = 1
        else:
            cnt += 1

    groups.append(cnt)

    return np.array(groups)


@beartype
def polars_left_merge(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    common_columns = list(set(df1.columns) & set(df2.columns))
    return df1.join(df2, on=common_columns, how='left')


@beartype
def polars_outer_merge(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    common_columns = list(set(df1.columns) & set(df2.columns))
    return df1.join(df2, on=common_columns, how='outer')


@beartype
def cast_to_float32(pl_frame: pl.DataFrame) -> pl.DataFrame:
    float_columns = [
        col for col in pl_frame.columns
        if pl_frame[col].dtype == pl.Float64 or (pl_frame[col].dtype == pl.Int64 and pl_frame[col].null_count() > 0)
    ]
    return pl_frame.with_columns([pl.col(col).cast(pl.Float32) for col in float_columns])


@beartype
def get_df_name(df: pd.DataFrame) -> Optional[str]:
    try:
        return df.name
    except AttributeError:
        return None


@beartype
def flatten(data: Iterable[Iterable[int]]) -> Iterable:
    return (el for inner in data for el in inner)


@beartype
def get_sequential_index(recs: Iterable[Iterable[Any]]) -> np.ndarray:
    return np.repeat(range(len(recs)), [len(r) for r in recs])


@beartype
def create_targets_df(data: Sessions) -> pl.DataFrame:
    df = pl.DataFrame({
        'index': get_sequential_index(data),
        'item_id': list(flatten(data)),
        'target': [1 / (rank + 1) for rec in data for rank, _ in enumerate(rec)],
    })
    return df


@beartype
def create_df_from_sessions(sessions: Sessions):
    df = pd.DataFrame({
        'user_id': get_sequential_index(sessions),
        'item_id': flatten(sessions),
        'rank': flatten(reversed(range(1, len(r) + 1)) for r in sessions)
    }, dtype='int32')
    return df


@beartype
def plot_user_activity(list_users: List[int], df: pd.DataFrame) -> None:
    actions = ["clicks", "carts", "orders"]
    dict_type_action = dict()
    for action in actions:
        dict_type_action[action] = dict()

    dict_type_action["clicks"]["color"] = "b"
    dict_type_action["carts"]["color"] = "orange"
    dict_type_action["orders"]["color"] = "r"

    dict_type_action["clicks"]["alpha"] = 0.3
    dict_type_action["carts"]["alpha"] = 0.7
    dict_type_action["orders"]["alpha"] = 0.9

    offset = 0.5
    df_name = get_df_name(df)
    if df_name == "df_train":
        x = pd.date_range("2022-08-01", "2022-08-28")
    elif df_name == "df_test":
        x = pd.date_range("2022-08-29", "2022-09-04")
    else:
        x = pd.date_range("2022-08-01", "2022-09-04")
    x = [str(d).split(" ")[0] for d in x]
    map_dates = {x[i]: i for i in range(len(x))}
    points_sundays = [point for point in range(len(x)) if (point + 1) % 7 == 0]

    for user in list_users:
        df_cur = df[df["user"] == user]
        df_cur["date_day"] = df_cur["date_day"].astype(str)  
        n_sessions = df_cur['n_session'].max() + 1

        plt.figure(figsize=(12, 7))
        plt.xticks(range(len(x)), x, rotation=90)
        plt.yticks(range(25))

        for point in range(25):
            plt.axhline(y=point, color="gray", linestyle="-", alpha=0.1)
        for point in range(len(x)):
            plt.axvline(x=point + offset, color="gray", linestyle="-", alpha=0.1)
            plt.axvline(x=point - offset, color="gray", linestyle="-", alpha=0.1)
        for point in points_sundays:
            plt.axvline(x=point + offset, color="r", linestyle="-", alpha=0.3)
            plt.axvline(x=point - offset, color="r", linestyle="-", alpha=0.3)
            plt.axvline(x=point - offset - 1, color="r", linestyle="-", alpha=0.3)

        dict_count_user_actions = {action: 0 for action in dict_type_action}

        for action in actions:
            df_cur_action = df_cur[df_cur["type"] == action]
            n_actions = len(df_cur_action)
            dict_count_user_actions[action] = n_actions

            xx = np.random.uniform(-0.3, 0.3, n_actions)
            yy = np.random.uniform(-0.3, 0.3, n_actions)

            plt.scatter(
                df_cur_action["date_day"].map(map_dates) + xx,
                df_cur_action["hour"] + yy,
                c=dict_type_action[action]["color"],
                alpha=dict_type_action[action]["alpha"],
                s=10,
                label=action,
            )
        plt.legend(
            loc="upper right", bbox_to_anchor=(1.1, 1.0), ncol=1, fancybox=True, shadow=True
        )

        stats_str = ''
        for action in dict_count_user_actions:
            stats_str += f'{dict_count_user_actions[action]} {action}, '
        stats_str = stats_str[:-2]
        title = f'User {user} has {n_sessions} real sessions with {sum(dict_count_user_actions.values())} total actions: {stats_str}'
        plt.title(title, size=14)
        plt.show()


@beartype
def make_date_stats(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime((df["ts"] + 1000 * 60 * 60 * 2) * 1e6)
    df['date_day'] = df['date'].dt.to_period("d")#.astype(str)
    df['n_day'] = df['date'].dt.day.astype('int8')
    df['n_week'] = df['date'].dt.isocalendar().week.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')
    df['hour'] = df['date'].dt.hour.astype('int8')
    return df


@beartype
def split_list_randomly(l: List, seed: int = 1337) -> Tuple[list, list]:
    np.random.seed(seed)
    len_list = len(l)
    pos_split = np.random.randint(low=1, high=len_list)
    part_one = l[:pos_split]
    part_two = l[pos_split:]
    return (part_one, part_two)


@beartype
def recall_vals_one_row(candidates: List[Union[int, np.int64]], target: Union[int, np.int64]) -> bool:
    return target in set(candidates)


@beartype
def recall_one_val(candidates: List[List[Union[int, np.int64]]], target: List[Union[int, np.int64]]) -> float:
    total_len = len(candidates)
    total_hits = 0
    for i in range(total_len):
        total_hits += recall_vals_one_row(candidates[i], target[i])
    return total_hits / total_len


@beartype
def truncate_list(l: List, max_len: int, from_start: bool = False):
    if not from_start:
        return l[-max_len:]  # last max_len values
    return l[:max_len]  # first max_len values


@beartype
def recall_several_targets_one_row(
    candidates: List[Union[int, np.int64]], 
    targets: List[Union[int, np.int64]], 
    len_candidates_required : int = 20,
    force_truncate_candidates : bool = False
) -> float:

    if force_truncate_candidates:
        candidates = [candidates_cur[:len_candidates_required] for candidates_cur in candidates]

    return len(set(candidates) & set(targets)) / min(len_candidates_required, len(targets)) 


@beartype
def recall_several_targets(
    candidates: List[List[Union[int, np.int64]]], 
    targets: List[List[Union[int, np.int64]]], 
    len_candidates_required : int = 20,
    force_truncate_candidates : bool = False
) -> float:
    assert len(candidates) == len(targets)

    if force_truncate_candidates:
        candidates = [candidates_cur[:len_candidates_required] for candidates_cur in candidates]

    denominator = sum(
        [min(len_candidates_required, len(target)) for target in targets]
    )
    numerator = sum(
        [len(set(candidates[i]) & set(targets[i])) for i in range(len(targets))]
    )
    return numerator / denominator


@beartype
def flatten_list(l: List):
    return [item for sublist in l for item in sublist]


@beartype
def split_data(df: pd.DataFrame, validation_cut: int, test_users: Union[pl.internals.series.series.Series, List], config: Any, new_split: bool = False):
    if not new_split:
        df_val = df[
            (df[config.col_num_action] <= validation_cut)
            & (df[config.col_user].isin(test_users))
        ]

        df_val_seq = df[
            (df[config.col_num_action] > validation_cut)
            & (df[config.col_user].isin(test_users))
        ]

        df_train = df.drop(df_val.index)
    else:
        df = pl.from_pandas(df)

        df_for_split = df.filter(pl.col(config.col_user).is_in(test_users))
        df_for_split = df_for_split.join(df_for_split.groupby([config.col_user]).count(), on=[config.col_user])

        df_for_split_less = df_for_split.filter((pl.col('count') <= validation_cut) & (pl.col('count') > 1))
        df_for_split_more = df_for_split.filter(pl.col('count') > validation_cut)

        test_less = df_for_split_less.filter(pl.col(config.col_num_action) == 1)
        test_more = df_for_split_more.filter(pl.col(config.col_num_action) <= validation_cut)

        df_val = pl.concat([test_less, test_more])

        df = df.join(df_val, on=[config.col_user, config.col_aid, config.col_ts, config.col_type, config.col_num_action], how='left')

        df_train = df.filter(pl.col('count').is_null())
        df_train = df_train.drop(columns=['count'])

        df_val = df_val.drop(columns=['count'])

        # val_users = df_val[config.col_user].unique()
        # df_val_seq = df_train.filter(pl.col(config.col_user).is_in(val_users))
        df_val_seq = df_train.filter(pl.col(config.col_user).is_in(test_users))

        df_train = df_train.to_pandas()
        df_val = df_val.to_pandas()
        df_val_seq = df_val_seq.to_pandas()
    return df_train, df_val, df_val_seq


@beartype
def get_train_val_dfs(
    config: Any, 
    count_aid_trs: int = 20, 
    validation_cut: int = 3,
    do_ordinal_encoder_transform: bool = True,
    drop_pad: bool = False,
    new_split: bool = False
):
    PATH_DATA = config.path_data
    df_train = pl.read_parquet(PATH_DATA + 'df_train_n_session.parquet')
    df_test = pl.read_parquet(PATH_DATA + 'df_test_n_session.parquet')
    test_users = df_test[config.col_user].unique()
    df = pl.concat([df_train, df_test])
    df = df.sort(config.col_ts, reverse=True)

    # df = df.to_pandas()
    # df[config.col_num_action] = df.groupby(config.col_user)[config.col_aid].cumcount()
    # df = pl.from_pandas(df)

    df = df.with_column(pl.col(config.col_ts).
               rank(method="ordinal", reverse=True).
               over([config.col_user]).alias(config.col_num_action))

    df = df.with_columns(
        [
            pl.col(config.col_num_action)
            .cast(pl.Int32, strict=True)
            .alias(config.col_num_action),
            pl.col(config.col_n_session)
            .cast(pl.Int32, strict=True)
            .alias(config.col_n_session),
        ]
    )

    del df_train, df_test
    gc.collect()

    counts_aid = df.groupby([config.col_aid]).count().sort("count", reverse=True)

    df = df.to_pandas()
    X_train, X_val, X_val_seq = split_data(df, validation_cut=validation_cut, test_users=test_users, config=config, new_split=new_split)
    del df
    gc.collect()

    counts_aid_filtered = counts_aid.filter(pl.col("count") >= count_aid_trs).to_pandas()
    pad = len(counts_aid_filtered)

    ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=pad)
    ordinal_encoder.fit(counts_aid_filtered[config.col_aid].values.reshape(-1, 1))

    if do_ordinal_encoder_transform:
        X_train[config.col_aid] = ordinal_encoder.transform(
            X_train[config.col_aid].values.reshape(-1, 1)
        ).astype(int)

        X_val_seq[config.col_aid] = ordinal_encoder.transform(
            X_val_seq[config.col_aid].values.reshape(-1, 1)
        ).astype(int)

        if drop_pad:
            X_train = X_train[X_train[config.col_aid] != pad]
            X_val_seq = X_val_seq[X_val_seq[config.col_aid] != pad]

    X_val = pl.from_pandas(X_val)
    X_val_seq = pl.from_pandas(X_val_seq)
    X_train = pl.from_pandas(X_train)

    return X_train, X_val, X_val_seq, ordinal_encoder


def dfs_interactions_to_list(
    config,
    X_train,
    X_val,
    X_val_seq, 
    ordinal_encoder,
    return_only_val: bool = False,
    truncate_train: bool = False,
    truncate_val: bool = False,
    return_just_splitted: bool = False,
):
    pad = ordinal_encoder.unknown_value
    X_val = X_val.sort(config.col_ts)

    X_val_seq = X_val_seq.sort(config.col_ts)

    X_val = (
        X_val.groupby([config.col_user])
        .agg(
            [
                pl.col(config.col_aid).list(),
            ]
        )
    )
    
    X_val_seq = (
        X_val_seq.groupby([config.col_user])
        .agg(
            [
                pl.col(config.col_aid).list(),
            ]
        )
    )

    if truncate_val:
        X_val_seq = X_val_seq.to_pandas()
        X_val_seq[config.col_train_seq] = X_val_seq[config.col_aid].progress_apply(
            lambda x: truncate_list(l=list(x), max_len=config.max_len, from_start=False)
        )
        X_val_seq = pl.from_pandas(X_val_seq)
    else:
        X_val_seq = X_val_seq.with_column(pl.col(config.col_aid).alias(config.col_train_seq))

    X_val = X_val.join(
        X_val_seq[[config.col_user, config.col_train_seq]], how="inner", on=config.col_user
    )
    X_val = X_val.rename({config.col_aid: config.col_target})


    if not return_only_val:
        X_train = X_train.sort(config.col_ts)

        X_train = (
            X_train.groupby([config.col_user])
            .agg(
                [
                    pl.col(config.col_aid).list(),
                ]
            ).filter(pl.col(config.col_aid).apply(lambda x: len(x)) >= 2)
        )
        if not return_just_splitted:
            X_train = X_train.to_pandas()
            X_train[config.col_train_seq] = X_train[config.col_aid].progress_apply(lambda x: x[:-1])
            X_train[config.col_target] = X_train[config.col_aid].progress_apply(lambda x: x[-1])
            X_train = pl.from_pandas(X_train)
            X_train = X_train.filter(pl.col(config.col_target) != pad)
            if truncate_train:
                X_train = X_train.to_pandas()
                X_train[config.col_train_seq] = X_train[config.col_aid].progress_apply(
                    lambda x: truncate_list(l=list(x), max_len=config.max_len, from_start=False)
                )
                X_train = pl.from_pandas(X_train)
    else:
        X_train = None

    return X_train, X_val, X_val_seq, ordinal_encoder
    
###

def get_df_from_file(path, user_id_start=0):
  user_id = user_id_start
  df_train = []
  with open(path) as f:
      lines = f.readlines()
      
      for line in tqdm(lines):
          tracks = line.strip().split(' ')
          tracks = list(map(int, tracks))
          df_train.append(np.vstack([len(tracks)*[user_id], tracks]).T)
          user_id += 1

  df_train = pd.DataFrame(np.vstack(df_train), columns=['customer_id', 'article_id'])
  df_train = df_train.astype({'customer_id': np.int32, 'article_id': np.int32})
  return df_train


# def split_data(df, validation_cut=7):
#     """ Split a pandas dataframe into training and validation data, using <<validation_days>>
#     """
#     df_train = df[df['num_action'] > validation_cut]
#     df_val = df[df['num_action'] <= validation_cut]
#     return df_train, df_val


def makesub(df, df_test, name='submission'):
   # df = df.sort_values('pred')
    sub = pd.read_csv('data/sample_submission.csv')
    dummy_list_2_week = df_test['article_id'].value_counts().keys()[:20]
    df_pred = df.groupby(['customer_id'])['article_id'].agg(list)
    df_pred = df_pred.apply(lambda x: x+ list(dummy_list_2_week))
    df_pred = df_pred.sort_index()
    df_pred = df_pred.apply(lambda best_tracks: ' '.join([str(x) for x in best_tracks[:20]]))
    df_pred.index = list(map(str, list(df_pred.index)))
    clicks_pred_df = df_pred.add_suffix("_clicks").to_frame('labels').reset_index()
    orders_pred_df = df_pred.add_suffix("_orders").to_frame('labels').reset_index()
    carts_pred_df = df_pred.add_suffix("_carts").to_frame('labels').reset_index()
    pred_df = pd.concat(
        [clicks_pred_df, orders_pred_df, carts_pred_df]
    )
    pred_df.columns = ["session_type", "labels"]
    add_sub = pd.DataFrame(set(sub['session_type']) - set(pred_df['session_type']), columns =['session_type'])
    add_sub['labels'] = ' '.join([str(x) for x in dummy_list_2_week[:20]])
    pred_df = pd.concat(
        [pred_df, add_sub]
    )
    pred_df = pd.merge(sub[['session_type']], pred_df, on=['session_type'])
    pred_df.to_csv(f"sub/{name}.csv", index=False)


def sub_for_metrics(df, df_traks, test_users):
   # df = df.sort_values('pred')
    dummy_list_2_week = df_traks[df_traks['t_dat']>df_traks['t_dat'].max()-pd.Timedelta(days=7)]['article_id'].value_counts().keys()[:20]
    df_pred = df.groupby(['customer_id'])['article_id'].agg(list)
    df_pred = df_pred.apply(lambda x: x+ list(dummy_list_2_week))
    df_pred = df_pred.sort_index()
    df_pred = df_pred.apply(lambda best_tracks: ' '.join([str(x) for x in best_tracks[:20]]))
    df_pred.index = list(map(str, list(df_pred.index)))
    test_users = list(map(str, list(test_users)))
    add_sub = set(test_users) - set(df_pred.index)
    clicks_pred_df = df_pred.add_suffix("_clicks").to_frame('labels').reset_index()
    orders_pred_df = df_pred.add_suffix("_orders").to_frame('labels').reset_index()
    carts_pred_df = df_pred.add_suffix("_carts").to_frame('labels').reset_index()
    pred_df = pd.concat(
        [clicks_pred_df, orders_pred_df, carts_pred_df]
    )
    pred_df.columns = ["session_type", "labels"]
    add_sub = pd.DataFrame(add_sub, columns =['session'])
    add_sub['labels'] = ' '.join([str(x) for x in dummy_list_2_week[:20]])

    add_sub['session_type'] = add_sub['session'] + '_clicks'
    clicks_pred_df = add_sub[['session_type', 'labels']]
    add_sub['session_type'] = add_sub['session'] + '_orders'
    orders_pred_df = add_sub[['session_type', 'labels']]
    add_sub['session_type'] = add_sub['session'] + '_carts'
    carts_pred_df = add_sub[['session_type', 'labels']]

    add_sub = pd.concat(
        [clicks_pred_df, orders_pred_df, carts_pred_df]
    )

    pred_df = pd.concat(
        [pred_df, add_sub]
    )
    return pred_df


def get_metric(submission, df_test_traks):
    submission['session'] = submission.session_type.apply(lambda x: int(x.split('_')[0]))
    submission['type'] = submission.session_type.apply(lambda x: x.split('_')[1])
    submission.labels = submission.labels.apply(lambda x: [int(i) for i in x.split(' ')][:20])
    df_test_traks = df_test_traks.rename({'customer_id': 'session' , 'article_id': 'aid' , 'ts': 't_dat'}, axis=1)

    #df_test_traks.type = df_test_traks.type.map(lambda idx: id2type[idx])
    ground_truth = df_test_traks.groupby(['session', 'type'])['aid'].apply(list)
    ground_truth = ground_truth.reset_index().rename(columns={'aid': 'labels'})
    ground_truth.loc[ground_truth.type == 'clicks', 'labels'] = ground_truth.loc[ground_truth.type == 'clicks', 'labels'].str[:1]

    submission_with_gt = submission.merge(ground_truth[['session', 'type', 'labels']], how='left', on=['session', 'type'])
    submission_with_gt = submission_with_gt[~submission_with_gt.labels_y.isna()]
    submission_with_gt['hits'] = submission_with_gt.apply(lambda df: len(set(df.labels_x).intersection(set(df.labels_y))), axis=1)
    submission_with_gt['gt_count'] = submission_with_gt.labels_y.str.len().clip(0,20)

    recall_per_type = submission_with_gt.groupby(['type'])['hits'].sum() / submission_with_gt.groupby(['type'])['gt_count'].sum() 
    local_validation_score = (recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()
    print(f'Local validation score: {local_validation_score}')