from collections import Counter, defaultdict

import gc
import polars as pl
import pandas as pd
from numpy import ndarray, int64, repeat
from beartype import beartype
from tqdm import tqdm

from typing import Union, List, Tuple, Dict
Session = Union[List[int], ndarray]
Sessions = List[Session]
Int = Union[int, int64]


@beartype
def get_sequential_index(recs: List[List[Tuple[Union[Int, float], Union[Int, float]]]]) -> ndarray:
    return repeat(range(len(recs)), [len(r) for r in recs])


class CoocurenceRecommender:
    @beartype
    def __init__(
            self,
            top_pairs: Dict[int, List[Tuple[int, int]]],
            item_index: int = 0,
            num_recs: int = 100,
            exclude_seen_items: bool = False,
    ):
        self.top_pairs = top_pairs
        self.item_index = item_index
        self.num_recs = num_recs
        self.exclude_seen_items = exclude_seen_items

    @beartype
    def recommend(self, data: Sessions) -> pl.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in tqdm(data, desc='predicting')
        ]

        return pl.DataFrame({
            'index': get_sequential_index(recs),
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
            return [item for item in top_pair_items if item[0] not in us][:self.num_recs]

        return [item for item in top_pair_items][:self.num_recs]


class CoocurenceTripletRecommender:
    @beartype
    def __init__(
            self,
            top_triples: Dict[Tuple[int, int], List[Tuple[int, int]]],
            item_index: int = 0,
            num_recs: int = 100,
            exclude_seen_items: bool = False,
    ):
        self.top_triples = top_triples
        self.item_index = item_index
        self.num_recs = num_recs
        self.exclude_seen_items = exclude_seen_items

    @beartype
    def recommend(self, data: Sessions) -> pl.DataFrame:
        recs = [
            self._recommend_one_user(user_data)
            for user_data in tqdm(data, desc='predicting')
        ]

        return pl.DataFrame({
            'index': get_sequential_index(recs),
            'item_id': [item[0] for rec in recs for item in rec],
            f'cooc_triplet_score_{self.item_index}': [item[1] for rec in recs for item in rec],
            f'cooc_triplet_rank_{self.item_index}': [rnk + 1 for rec in recs for rnk, _ in enumerate(rec)],
        })

    @beartype
    def _recommend_one_user(self, user_data: Session):
        if len(user_data) >= 2:
            last_items = (user_data[-self.item_index - 2], user_data[-self.item_index - 1])
            top_triplet_items = self.top_triples.get(
                last_items,
                list()
            )
        else:
            top_triplet_items = list()

        if self.exclude_seen_items:
            us = set(user_data)
            return [item for item in top_triplet_items if item[0] not in us][:self.num_recs]

        return [item for item in top_triplet_items][:self.num_recs]


@beartype
def train_cooc_model(
    data: Sessions, 
    max_candidates: int = 100,
    normalize: bool = True,
    filtering_rare_keys_trs: int = 20,
) -> dict:

    pairs = defaultdict(Counter)
    top_pairs = dict()

    for record in tqdm(data, desc='extracting pairs'):
        for i in range(len(record) - 1):
            last, ans = record[i], record[i + 1]
            pairs[last][ans] += 1

    if filtering_rare_keys_trs > 1:
        for key in tqdm(list(pairs.keys()), desc='filtering rare keys'):
            sum_key = sum(pairs[key].values())
            if sum_key < filtering_rare_keys_trs:
                del pairs[key]

    if normalize:
        for key in tqdm(list(pairs.keys()), desc='normalizing'):
            sum_key = sum(pairs[key].values())
            for key_cur in pairs[key]:
                pairs[key][key_cur] /= sum_key

    for key in tqdm(list(pairs.keys()), desc='extracting max_candidates'):
        top_pairs[key] = pairs[key].most_common(max_candidates)
        del pairs[key]

    return top_pairs


@beartype
def train_triplet_model(
    data: Sessions,
    max_candidates: int = 100,
    normalize: bool = True,
    filtering_rare_keys_trs: int = 10,
) -> dict:

    triples = defaultdict(Counter)
    top_triples = dict()

    for record in tqdm(data, desc='extracting triplets'):
        for i in range(len(record) - 2):
            prev, last, ans = record[i], record[i + 1], record[i + 2]
            triples[(prev, last)][ans] += 1

    if filtering_rare_keys_trs > 1:
        for key in tqdm(list(triples.keys()), desc='filtering rare keys'):
            sum_key = sum(triples[key].values())
            if sum_key < filtering_rare_keys_trs:
                del triples[key]

    if normalize:
        for key in tqdm(list(triples.keys()), desc='normalizing'):
            sum_key = sum(triples[key].values())
            for key_cur in triples[key]:
                triples[key][key_cur] /= sum_key

    for key in tqdm(list(triples.keys()), desc='extracting max_candidates'):
        top_triples[key] = triples[key].most_common(max_candidates)
        del triples[key]

    return top_triples
