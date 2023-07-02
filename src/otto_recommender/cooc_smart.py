from collections import Counter, defaultdict
from _operator import itemgetter

import gc
import polars as pl
import pandas as pd
import numpy as np
from beartype import beartype
from tqdm.notebook import tqdm
from sklearn.utils.extmath import softmax
from more_itertools import chunked
import utils
from typing import Union, List, Tuple, Dict, Optional, Iterable
Session = Union[List[int], np.ndarray]
Sessions = List[Session]
Int = Union[int, np.int64]


@beartype
def create_cooc_stats(df: pd.DataFrame, min_rank_diff: int = -3, max_rank_diff: int = 3, items_per_item: int = 10) -> pl.DataFrame:
    cooc = df.merge(df, on=['user_id']).query('rank_x != rank_y')
    cooc = cooc[(cooc['rank_y'] - cooc['rank_x']).between(min_rank_diff, max_rank_diff)]
    groups = cooc.groupby(['item_id_x', 'item_id_y'])['user_id'].count().to_frame().reset_index()
    groups.columns = ['item_id', 'rec_item_id', 'count']
    groups = groups.sort_values(['item_id', 'count'], ascending=[True, False])
    groups['rnk'] = groups.groupby('item_id').cumcount() + 1
    groups_filtered = groups[groups['rnk'] <= items_per_item]
    del groups_filtered['rnk']
    return pl.from_pandas(groups_filtered)#.set_index(['item_id', 'rec_item_id'])


@beartype
def get_item_popularity_stats(df: pd.DataFrame, popularity_last_items: Optional[int] = None) -> pd.DataFrame:
    if popularity_last_items is not None:
        pop = df[df['rank'] <= popularity_last_items].groupby('item_id')['rank'].count().reset_index().sort_values('item_id')
    else:
        pop = df.groupby('item_id')['rank'].count().reset_index().sort_values('item_id')
    pop.columns = ['rec_item_id', 'item_popularity']
    return pop


@beartype
def create_cooc_stats_by_chunks(
        train_sessions: Sessions,
        chunk_size: int = 200_000,
        popularity_last_items: Optional[int] = 50,
        min_rank_diff: int = -3,
        max_rank_diff: int = 3,
        items_per_item: int = 50,
) -> pd.DataFrame:

    cooc_stats = list()
    print('creating df from sessions')
    df = utils.create_df_from_sessions(train_sessions)
    print('creating item popularity')
    item_popularity = get_item_popularity_stats(df, popularity_last_items)
    print('creating cooc_stats')
    users = df['user_id'].unique()

    for part in tqdm(range(len(users) // chunk_size + 1), desc='creating cooc_stats'):
        df_cur = df[df['user_id'].isin(set(users[part * chunk_size: (part + 1) * chunk_size]))]
        chunk_stats = create_cooc_stats(
            df_cur,
            min_rank_diff,
            max_rank_diff,
            items_per_item,
        )
        cooc_stats.append(chunk_stats)
    cooc_stats = pl.concat(cooc_stats)
    cooc_stats = cooc_stats.groupby(['item_id', 'rec_item_id']).agg(pl.col('count').sum()).to_pandas()
    cooc_stats = cooc_stats.sort_values(['item_id', 'count'], ascending=[True, False])
    cooc_stats['rnk'] = cooc_stats.groupby('item_id').cumcount() + 1
    cooc_stats = cooc_stats.merge(item_popularity)
    cooc_stats['cooc_norm'] = cooc_stats['count'] / cooc_stats['item_popularity']
    del cooc_stats['item_popularity']

    cooc_stats.columns = ['item_id', 'rec_item_id', 'cooc_count', 'cooc_rnk', 'cooc_norm']

    return cooc_stats


@beartype
def get_smart_cooc_candidates(
        user_history_chunk: pd.DataFrame,
        cooc_stats: pd.DataFrame,
        history_last_items: int,
        max_candidates: int = 50
) -> pd.DataFrame:
    recent_history = user_history_chunk[user_history_chunk['rank'] <= history_last_items]
    user_history_cooc = (
        recent_history
        .merge(cooc_stats)
        .groupby(['user_id', 'rec_item_id'])[['cooc_count']]
        .agg(['sum'])
        .reset_index()
    )
    user_history_cooc.columns = ['user_id', 'item_id', 'cooc_smart_score']  # 'cooc_count_sum'
    cooc_candidates = user_history_cooc.merge(user_history_chunk, how='left')
    cooc_candidates = cooc_candidates[cooc_candidates['rank'].isnull()]
    cooc_candidates = cooc_candidates.sort_values(['user_id', 'cooc_smart_score'], ascending=[True, False])
    cooc_candidates['cooc_smart_rank'] = cooc_candidates.groupby('user_id')['item_id'].cumcount()

    return cooc_candidates[cooc_candidates['cooc_smart_rank'] <= max_candidates]


@beartype
def get_smart_cooc_candidates_by_chunks(
        val_or_test_history: pd.DataFrame,
        cooc_stats: pd.DataFrame,
        history_last_items: int = 1,
        max_candidates: int = 50,
        chunk_size: int = 1000,
) -> pd.DataFrame:
    num_users = val_or_test_history['user_id'].nunique()
    all_candidates = []
    for user_ids in tqdm(chunked(range(num_users), chunk_size), total=num_users//chunk_size + 1, desc='predicting'):
        user_history_chunk = val_or_test_history[val_or_test_history['user_id'].isin(set(user_ids))]
        candidates = get_smart_cooc_candidates(user_history_chunk, cooc_stats, history_last_items, max_candidates)
        all_candidates.append(candidates)

    return pd.concat(all_candidates).reset_index(drop=True)


@beartype
def cooc_stats_to_dict(
    cooc_stats: pd.DataFrame, 
    max_candidates: int = 100,
    filtering_rare_keys_trs: int = 0,
    chunk_size = 500_000
) -> dict:

    dict_cooc_stats = defaultdict(dict)
    top_pairs = dict()
    items = cooc_stats["item_id"].unique()

    for part_items in tqdm(chunked(range(len(items)), chunk_size), total=len(items)//chunk_size + 1, desc='transforming'):
        df_part = cooc_stats[['item_id', 'rec_item_id', "cooc_norm"]]
        df_part = df_part[df_part['item_id'].isin(set(part_items))]
        dict_cur = df_part.set_index(["item_id", "rec_item_id"])[["cooc_norm"]].to_dict()["cooc_norm"]
        for key in dict_cur:
            dict_cooc_stats[key[0]][key[1]] = dict_cur[key]

    if filtering_rare_keys_trs > 1:
        for key in tqdm(list(dict_cooc_stats.keys()), desc='filtering rare keys'):
            sum_key = sum(dict_cooc_stats[key].values())
            if sum_key < filtering_rare_keys_trs:
                del dict_cooc_stats[key]

    for key in tqdm(list(dict_cooc_stats.keys()), desc='extracting max_candidates'):
        top_pairs[key] = Counter(dict_cooc_stats[key]).most_common(max_candidates)
        del dict_cooc_stats[key]

    return top_pairs


class CoocurenceRecommender:
    @beartype
    def __init__(
            self,
            top_pairs: Dict[int, List[Tuple[int, int]]],
            item_index: int = 0,
            n_recs: int = 100,
            exclude_seen_items: bool = False,
    ):
        self.top_pairs = top_pairs
        self.item_index = item_index
        self.n_recs = n_recs
        self.exclude_seen_items = exclude_seen_items

    @beartype
    def recommend(self, data: Sessions, last_k_items: Optional[int] = None) -> pl.DataFrame:
        if not last_k_items:
            recs = [
                self._recommend_one_user(user_data)
                for user_data in tqdm(data, desc='predicting')
            ]
        else:
            weights = softmax([np.logspace(0.5, 1, last_k_items, base=last_k_items / 1.5, endpoint=True)])[0]
            recs = [
                self._recommend_one_user_from_k_last_items(user_data, weights, k_last=last_k_items)
                for user_data in tqdm(data, desc='predicting')
                # for user_data in data
            ]

        return pl.DataFrame({
            'index': utils.get_sequential_index(recs),
            'item_id': [item[0] for rec in recs for item in rec],
            f'cooc_score_{self.item_index}': [item[1] for rec in recs for item in rec],
            f'cooc_rank_{self.item_index}': [rnk + 1 for rec in recs for rnk, _ in enumerate(rec)],
        })

    @beartype
    def _recommend_one_user(self, user_data: Session) -> List:
        if len(user_data) <= self.item_index:
            return []
        top_pair_items = self.top_pairs.get(user_data[-self.item_index - 1], [])

        if self.exclude_seen_items:
            us = set(user_data)
            return [item for item in top_pair_items if item[0] not in us][:self.n_recs]

        return [item for item in top_pair_items][:self.n_recs]

    @beartype
    def _recommend_one_user_from_k_last_items(
        self, 
        user_data: Session, 
        weights: Iterable,
        k_last: int = 5
    ) -> List:
        user_recs = defaultdict(float)
        items_user_to_predict = user_data[-k_last:]
        len_items_user_to_predict = len(items_user_to_predict)
        # next_items = list(items_user_to_predict)

        for i in range(len(items_user_to_predict)):
            weights_cur = weights[-min(k_last, len_items_user_to_predict):]
            # next_items.pop(0)

            item = items_user_to_predict[i]
            weight = weights_cur[i]

            recs_scores = self.top_pairs.get(item, [])
            for rec, score in recs_scores:
                # if i == (len_items_user_to_predict - 1) or rec not in next_items:
                user_recs[rec] += score * weight

        if self.exclude_seen_items:
            seen = set(user_data)
            return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True) if x not in seen][:self.n_recs]

        return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True)][:self.n_recs]
