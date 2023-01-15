import numpy as np
import polars as pl
from beartype import beartype
from tqdm.notebook import tqdm

from typing import Union, List, Tuple, Dict, Iterable, Any, Optional
Int = Union[int, np.int64]
Float = Union[float, np.float64]
Session = Union[List[Int], np.ndarray]
Sessions = List[Session]


@beartype
def get_sequential_index(recs: List[List[Any]]) -> np.ndarray:
    return np.repeat(range(len(recs)), [len(r) for r in recs])


class LastItemsRecommender:

    def __init__(self):
        pass

    @beartype
    def recommend(self, data: Sessions, n_recs: int = 100) -> pl.DataFrame:
        recs = [
            self._recommend_one_user(user_data, n_recs)
            for user_data in tqdm(data, desc='predicting')
            # for user_data in data
        ]

        return pl.DataFrame({
            'index': get_sequential_index(recs),
            'item_id': [item for rec in recs for item in rec],
            # 'similar_score': [item[1] for rec in recs for item in rec],
            'similar_rank': [rnk + 1 for rec in recs for rnk, _ in enumerate(rec)],
        })

    @beartype
    def _recommend_one_user(self, user_data: Session, n_recs: int = 100) -> List:
        return list(dict.fromkeys(user_data[::-1]))[:n_recs]
