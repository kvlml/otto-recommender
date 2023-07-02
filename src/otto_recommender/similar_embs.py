from _operator import itemgetter

from collections import defaultdict

import numpy as np
import polars as pl
from beartype import beartype
from tqdm.notebook import tqdm
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import softmax

from typing import Union, List, Tuple, Dict, Iterable, Any, Optional, Iterable
Int = Union[int, np.int64]
Float = Union[float, np.float64]
Session = Union[List[Int], np.ndarray]
Sessions = List[Session]


@beartype
def flatten(data: Iterable[List[int]]) -> Iterable:
    return (el for inner in data for el in inner)


@beartype
def get_sequential_index(recs: List[List[Any]]) -> np.ndarray:
    return np.repeat(range(len(recs)), [len(r) for r in recs])


@beartype
def spm(data: Sessions, n_items: int) -> csr_matrix:
    items_count = [len(items) for items in data]
    ones = np.ones(sum(items_count))
    user_ids = np.repeat(range(len(data)), items_count)
    item_ids = list(flatten(data))
    mat = csr_matrix(
        (ones, (user_ids, item_ids)),
        shape=(len(data), n_items),
    )

    return mat


class SimilarEmbeddingsRecommender:

    def __init__(self, dict_sims: dict):
        self.dict_sims = dict_sims

    @beartype
    def recommend(self, data: Sessions, n_recs: int = 20, exclude_seen_items: bool = False, last_k_items: Optional[int] = None) -> pl.DataFrame:
        if last_k_items is None:
            recs = [
                self._recommend_one_user(user_data, n_recs, exclude_seen_items)
                for user_data in tqdm(data, desc='predicting')
                # for user_data in data
            ]
        else:
            weights = softmax([np.logspace(0.5, 1, last_k_items, base=last_k_items / 1.5, endpoint=True)])[0]
            recs = [
                self._recommend_one_user_from_k_last_items(user_data, weights, n_recs, exclude_seen_items, k_last=last_k_items)
                for user_data in tqdm(data, desc='predicting')
                # for user_data in data
            ]

        return pl.DataFrame({
            'index': get_sequential_index(recs),
            'item_id': [item[0] for rec in recs for item in rec],
            'similar_score': [item[1] for rec in recs for item in rec],
            'similar_rank': [rnk + 1 for rec in recs for rnk, _ in enumerate(rec)],
        })

    @beartype
    def _recommend_one_user(self, user_data: Session, n_recs: int = 100, exclude_seen_items: bool = False) -> List:
        user_recs = defaultdict(float)
        for item in user_data[:-2:-1]:
            recs, scores = self.dict_sims.get(item, ([], []))
            for rec, score in zip(recs, scores):
                user_recs[rec] += score

        if exclude_seen_items:
            seen = set(user_data)
            return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True) if x not in seen]

        return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True)]


    @beartype
    def _recommend_one_user_from_k_last_items(
        self, 
        user_data: Session, 
        weights: Iterable,
        n_recs: int = 100, 
        exclude_seen_items: bool = False, 
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

            recs, scores = self.dict_sims.get(item, ([], []))
            for rec, score in zip(recs, scores):
                # if i == (len_items_user_to_predict - 1) or rec not in next_items:
                user_recs[rec] += score * weight

        if exclude_seen_items:
            seen = set(user_data)
            return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True) if x not in seen][:n_recs]

        return [x for x in sorted(user_recs.items(), key=itemgetter(1), reverse=True)][:n_recs]


@beartype
def train_similar_model(data: Sessions, n_items: int, recommeder_cls: str, max_neighbours: int = 100):
    train_matrix = spm(data, n_items)
    recommender = recommenders[recommeder_cls](K=max_neighbours)
    recommender.fit(train_matrix)

    return recommender
