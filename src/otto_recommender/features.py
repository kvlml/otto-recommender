import pandas as pd
import numpy as np
import polars as pl
from sklearn.preprocessing import OrdinalEncoder
from tqdm.notebook import tqdm
tqdm.pandas()
import os
import gc
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Dict, Iterable, Any, Optional
from beartype import beartype
import warnings
from pandarallel import pandarallel
from numba import jit, njit, prange
import utils

Int = Union[int, np.int64]
Float = Union[float, np.float64]
Session = Union[List[Int], np.ndarray]
Sessions = List[Session]

# pandarallel.initialize(nb_workers=30, progress_bar=True, use_memory_fs=False)

warnings.filterwarnings("ignore")


@beartype
def get_item_stats_per_day(df: pl.DataFrame, max_rank: int = 50) -> pl.DataFrame:
    cols_to_gp_all = ["n_day", "aid"]
    cols_to_gp_wo_aid = ["n_day"]

    df_stats = df.groupby(cols_to_gp_all).count().with_column(
        pl.col("count")
        .rank(method="ordinal", reverse=True)
        .over(cols_to_gp_wo_aid)
        .alias("rank")
    ).filter(pl.col("rank") <= max_rank).sort("rank")

    df_stats = utils.cast_to_float32(df_stats)
    return df_stats


@beartype
def make_rank_features(df: pl.DataFrame) -> pl.DataFrame:
    cols_rank = [col for col in df.columns if col.endswith('_rank')]

    df = df.with_column(df[cols_rank].mean(axis=1).alias("rank_avg"))
    df = df.with_column(df[cols_rank].min(axis=1).alias("rank_min"))
    df = df.with_column(df[cols_rank].max(axis=1).alias("rank_max"))
    df = df.with_columns(pl.Series(df.select(cols_rank).to_pandas().std(axis=1)).alias('rank_std'))
    df = df.with_columns(pl.Series(df.select(cols_rank).to_pandas().median(axis=1)).alias('rank_median'))
    df = df.with_columns(pl.Series(df.select(cols_rank).to_pandas().isnull().sum(axis=1)).alias('rank_cnt_nulls'))

    cols_rank.extend(['rank_min', 'rank_max', 'rank_cnt_nulls'])
    for col in cols_rank:
        df = df.with_column(pl.col(col).fill_null(-99))
        df = df.with_column(
            pl.col(col).cast(pl.Int8, strict=False).alias(col)
        )
    df = utils.cast_to_float32(df)
    return df


@beartype
def inverse_rank_by_row(row: List) -> float:
    row = [val for val in row if val > 0]
    res = 0.0
    for val in row:
        res += 1 / val
    return res


@beartype
def make_inverse_rank_features(df: pl.DataFrame) -> pl.DataFrame:
    cols_rank = [col for col in df.columns if col.endswith('_rank')]
    # cols_rank = [
    #     'rnn_rank',
    #     'weak_nns_rank',
    #     'cooc_rank',
    #     'bm25_rank',
    #     'tfidf_rank',
    #     'cosine_rank',
    #     'last_items_rank'
    # ]
    # dict_weights = dict()
    # dict_weights['rnn_rank'] = 0.9
    # dict_weights['weak_nns_rank'] = 0.4
    # dict_weights['cooc_rank'] = 0.7
    # dict_weights['bm25_rank'] = 1.0
    # dict_weights['tfidf_rank'] = 1.0
    # dict_weights['cosine_rank'] = 1.0
    # dict_weights['last_items_rank'] = 2.0

    cols_inverse = list()

    for col in cols_rank:
        col_name = f'inverse_{col}'
        cols_inverse.append(col_name)
        df = df.with_column((pl.when(pl.col(col) > 0).then(1.0 / pl.col(col)).otherwise(0.0)).alias(col_name))
        # df = df.with_column((pl.when(pl.col(col) > 0).then(dict_weights[col] / pl.col(col)).otherwise(0.0)).alias(col_name))

    df = df.with_column(df[cols_inverse].sum(axis=1).alias("inverse_rank_level_1"))

    for col in cols_inverse:
        df = df.drop(col)
    df = utils.cast_to_float32(df)
    return df


@beartype
def get_user_item_stats(df_history: pl.DataFrame, col_user: str = 'session', col_item: str = 'item_id') -> dict:
    # df_history = test or val
    dict_df_history_cnt_interactions_types = dict()
    dict_df_history_cnt_interactions_types['total'] = df_history.groupby([col_user, col_item]).count()
    dict_df_history_cnt_interactions_types['total'] = dict_df_history_cnt_interactions_types['total'].rename({'count': 'cnt_interactions_total'})

    for interaction in ["clicks", "carts", "orders"]:
        dict_df_history_cnt_interactions_types[interaction] = df_history.filter(pl.col("type") == interaction).groupby([col_user, col_item]).count()
        dict_df_history_cnt_interactions_types[interaction] = dict_df_history_cnt_interactions_types[interaction].rename(
            {'count': f'cnt_interactions_{interaction}'}
        )

    return dict_df_history_cnt_interactions_types


@beartype
def make_user_item_features(
    df: pl.DataFrame, 
    user_item_stats: dict, 
    col_user: str = 'session', 
    col_item: str = 'item_id',
) -> pl.DataFrame:

    for key in user_item_stats:
        col_name = f'cnt_interactions_{key}'
        df = df.join(user_item_stats[key], on=[col_user, col_item], how='left')
        df = df.with_column(pl.col(col_name).fill_null(0))
        df = df.with_column(pl.col(col_name).cast(pl.UInt8))
    for pair in [('clicks', 'carts'), ('clicks', 'orders'), ('carts', 'orders')]:
        df = df.with_column((pl.col(f'cnt_interactions_{pair[1]}') / pl.col(f'cnt_interactions_{pair[0]}')).alias(f'ratio_user_{pair[0]}_to_{pair[1]}'))
    df = utils.cast_to_float32(df)
    return df


@beartype
def make_user_features(
    df: pl.DataFrame, 
    df_history: pl.DataFrame,
    col_user: str = 'session', 
    col_item: str = 'item_id',
) -> pl.DataFrame:
    # df_history = test or val
    cols_to_drop = list()
    for interaction in ["clicks", "carts", "orders"]:
        col_name = f'cnt_{interaction}_by_user'
        user_stats = df_history.filter(pl.col('type') == interaction).groupby(col_user).agg(pl.col('type').count().alias(col_name))
        df = df.join(user_stats, on=[col_user], how='left')
        cols_to_drop.append(col_name)
    for pair in [('clicks', 'carts'), ('clicks', 'orders'), ('carts', 'orders')]:
        df = df.with_column(
            (pl.col(f'cnt_{pair[1]}_by_user') / pl.col(f'cnt_{pair[0]}_by_user')).alias(f'ratio_user_total_{pair[0]}_to_{pair[1]}')
        )
    for col in cols_to_drop:
        df = df.drop(col)
    df = utils.cast_to_float32(df)
    return df


@beartype
def get_item_popularity_stats(df_history: pl.DataFrame, col_item: str = 'item_id') -> pl.DataFrame:
    # df_history : train or train+val
    item_popularity_stats = df_history.groupby(col_item).count()
    item_popularity_stats.columns = [col_item, 'item_popularity']

    perc_val_to_norm = np.percentile(list(item_popularity_stats['item_popularity']), q=95)
    item_popularity_stats = item_popularity_stats.with_column(
        (pl.col("item_popularity") / perc_val_to_norm)
        .alias("item_popularity")
    )

    # item_popularity_stats = item_popularity_stats.with_column(
    #     pl.col("item_popularity")
    #     .rank(method="ordinal", reverse=True)
    #     .alias("item_popularity_rnk")
    # )
    return item_popularity_stats


@beartype
def get_user_item_popularity_stats(
    df_history: pl.DataFrame, 
    item_popularity_stats: pl.DataFrame, 
    col_item: str = 'item_id', 
    col_user: str = 'session',
) -> dict:
    # df_history : val or test
    dict_user_item_popularity_stats = dict()

    user_item_popularity_stats = df_history.join(item_popularity_stats, on=col_item, how='left')
    dict_user_item_popularity_stats['total'] = user_item_popularity_stats.groupby([col_user]).agg(
        [
            pl.col("item_popularity").mean().alias("item_popularity_total_mean"),
            pl.col("item_popularity").median().alias("item_popularity_total_median"),
            pl.col("item_popularity").std().alias("item_popularity_total_std"),
            pl.col("item_popularity").min().alias("item_popularity_total_min"),
            pl.col("item_popularity").max().alias("item_popularity_total_max"),
            # pl.col("item_popularity_rnk").mean().alias("item_popularity_rnk_total_mean"),
            # pl.col("item_popularity_rnk").median().alias("item_popularity_rnk_total_median"),
            # pl.col("item_popularity_rnk").std().alias("item_popularity_rnk_total_std"),
            # pl.col("item_popularity_rnk").min().alias("item_popularity_rnk_total_min"),
            # pl.col("item_popularity_rnk").max().alias("item_popularity_rnk_total_max"),
        ]
    )

    for interaction in ["clicks", "carts", "orders"]:
        dict_user_item_popularity_stats[interaction] = user_item_popularity_stats.filter(pl.col("type") == interaction).groupby([col_user]).agg(
            [
                pl.col("item_popularity").mean().alias(f"item_popularity_{interaction}_mean"),
                pl.col("item_popularity").median().alias(f"item_popularity_{interaction}_median"),
                pl.col("item_popularity").std().alias(f"item_popularity_{interaction}_std"),
                pl.col("item_popularity").min().alias(f"item_popularity_{interaction}_min"),
                pl.col("item_popularity").max().alias(f"item_popularity_{interaction}_max"),
                # pl.col("item_popularity_rnk").mean().alias(f"item_popularity_rnk_{interaction}_mean"),
                # pl.col("item_popularity_rnk").median().alias(f"item_popularity_rnk_{interaction}_median"),
                # pl.col("item_popularity_rnk").std().alias(f"item_popularity_rnk_{interaction}_std"),
                # pl.col("item_popularity_rnk").min().alias(f"item_popularity_rnk_{interaction}_min"),
                # pl.col("item_popularity_rnk").max().alias(f"item_popularity_rnk_{interaction}_max"),
            ]
        )

    return dict_user_item_popularity_stats


@beartype
def make_user_item_popularity_features(
    df: pl.DataFrame, 
    item_popularity_stats: pl.DataFrame, 
    dict_user_item_popularity_stats: dict, 
    col_user: str = 'session', 
    col_item: str = 'item_id'
) -> pl.DataFrame:
    # df_history = test or val
    df = df.join(item_popularity_stats, on=[col_item], how='left')
    for key in dict_user_item_popularity_stats:
        df = df.join(dict_user_item_popularity_stats[key], on=[col_user], how='left')
        df = df.with_column(
            (pl.col("item_popularity") / pl.col(f"item_popularity_{key}_mean")).alias(
                f"item_popularity_to_item_popularity_{key}_mean"
            )
        )
        df = df.with_column(
            (pl.col("item_popularity") / pl.col(f"item_popularity_{key}_median")).alias(
                f"item_popularity_to_item_popularity_{key}_median"
            )
        )
    df = utils.cast_to_float32(df)
    return df


@beartype
def make_item_last_interation_features(
    df: pl.DataFrame,
    df_history: pl.DataFrame,
    col_user: str = 'session',
    col_item: str = 'item_id',
) -> pl.DataFrame:
    # df_history = test or val
    df_cur_moment = df_history.groupby([col_user]).agg(pl.col("ts").max())
    df_cur_moment = df_cur_moment.rename({'ts': 'cur_ts'})

    df_user_item_last_interaction = df_history.groupby([col_user, col_item, 'type']).agg(pl.col("ts").max())
    df_user_item_last_interaction = df_user_item_last_interaction.rename({'ts': 'last_interaction_ts'})

    df = df.join(df_cur_moment, on=[col_user], how='left')
    cols_new = list()
    for interaction in ["clicks", "carts", "orders"]:
        df = df.join(df_user_item_last_interaction.filter(pl.col('type') == interaction).drop('type'), on=[col_user, col_item], how='left')
        col_name = f'time_since_last_{interaction}_this_item_by_user'
        cols_new.append(col_name)
        df = df.with_column((pl.col('cur_ts') - pl.col('last_interaction_ts')).alias(col_name))
        df = df.with_column(pl.col(col_name).cast(pl.UInt32))
        for col in ['last_interaction_ts']:
            df = df.drop(col)

    df = df.with_column(df[cols_new].min(axis=1).alias("time_since_last_interaction_this_item_by_user"))
    df = df.with_column(pl.col('time_since_last_interaction_this_item_by_user').cast(pl.UInt32))
    df = df.drop('cur_ts')

    df_user_item_last_interaction_type = df_history.groupby([col_user, col_item]).agg(pl.col("type").last().alias('last_interaction_type'))
    df_user_item_last_interaction_type = df_user_item_last_interaction_type[[col_user, col_item, 'last_interaction_type']]
    dict_types = dict()
    dict_types["clicks"] = 0
    dict_types["carts"] = 1
    dict_types["orders"] = 2
    type_labels = pl.DataFrame(
        {"last_interaction_type": list(dict_types.keys()), "values": list(dict_types.values())}
    )
    df_user_item_last_interaction_type = (
        df_user_item_last_interaction_type.join(type_labels, on="last_interaction_type", how="left")
        .drop(columns=["last_interaction_type"])
        .rename({"values": "last_interaction_type"})
    )

    df = df.join(df_user_item_last_interaction_type, on=[col_user, col_item], how='left')

    df = df.with_column(pl.col('last_interaction_type').cast(pl.UInt8))
    df = utils.cast_to_float32(df)
    return df


@beartype
def make_item_interaction_to_interaction_features(
    df: pl.DataFrame,
    df_history: pl.DataFrame,
    col_user: str = 'session',
    col_item: str = 'item_id',
) -> pl.DataFrame:
    # df_history = train for valid or full for test
    dict_stats = dict()
    for interaction in ["clicks", "carts", "orders"]:
        dict_stats[interaction] = df_history.filter(pl.col("type") == interaction).groupby([col_item]).agg(
            pl.col(col_user).unique().count().alias(f'n_users_w_{interaction}_of_item')
        )
    for pair in [('clicks', 'carts'), ('clicks', 'orders'), ('carts', 'orders')]:
        df_stats_pair = dict_stats[pair[0]].join(dict_stats[pair[1]], how='left', on=col_item).fill_null(0)
        df_stats_pair = df_stats_pair.with_column(
            (pl.col(f'n_users_w_{pair[1]}_of_item') / pl.col(f'n_users_w_{pair[0]}_of_item')).alias(f'ratio_{pair[0]}_to_{pair[1]}')
        )
        for col in [f'n_users_w_{pair[0]}_of_item', f'n_users_w_{pair[1]}_of_item']:
            df_stats_pair = df_stats_pair.drop(col)
        df = df.join(df_stats_pair, on=[col_item], how='left')

    # test_week
    dict_stats = dict()
    for interaction in ["clicks", "carts", "orders"]:
        dict_stats[interaction] = df_history.filter(pl.col("n_week") == 35).filter(pl.col("type") == interaction).groupby([col_item]).agg(
            pl.col(col_user).unique().count().alias(f'n_users_w_{interaction}_of_item')
        )
    for pair in [('clicks', 'carts'), ('clicks', 'orders'), ('carts', 'orders')]:
        df_stats_pair = dict_stats[pair[0]].join(dict_stats[pair[1]], how='left', on=col_item).fill_null(0)
        df_stats_pair = df_stats_pair.with_column(
            (pl.col(f'n_users_w_{pair[1]}_of_item') / pl.col(f'n_users_w_{pair[0]}_of_item')).alias(f'ratio_{pair[0]}_to_{pair[1]}_test_week')
        )
        for col in [f'n_users_w_{pair[0]}_of_item', f'n_users_w_{pair[1]}_of_item']:
            df_stats_pair = df_stats_pair.drop(col)
        df = df.join(df_stats_pair, on=[col_item], how='left')

    df = utils.cast_to_float32(df)
    return df
