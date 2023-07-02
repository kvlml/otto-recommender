import numpy as np
import polars as pl
from beartype import beartype
from tqdm import tqdm
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from more_itertools import chunked

from typing import Union, List, Tuple, Dict, Iterable, Any
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


class AlsRecommender:

    def __init__(self, implicit_model: AlternatingLeastSquares):
        self.implicit_model = implicit_model

    @beartype
    def recommend(self, data: Sessions, n_recs: int = 100, chunk_size: int = 100_000) -> pl.DataFrame:
        n_items = self.implicit_model.item_factors.shape[0]
        all_recs = list()
        all_scores = list()
        total = len(data) // chunk_size if len(data) % chunk_size == 0 else len(data) // chunk_size + 1
        bar = tqdm(chunked(data, chunk_size), total=total)
        for batch_idx, batch in enumerate(bar):
            recs, scores = self.implicit_model.recommend(
                range(len(batch)),
                spm(batch, n_items),
                N=n_recs,
                recalculate_user=True,
                filter_already_liked_items=False,
            )
            all_recs.append(recs)
            all_scores.append(scores)

        recs, scores = np.vstack(all_recs).tolist(), np.vstack(all_scores)

        return pl.DataFrame({
            'index': get_sequential_index(recs),
            'item_id': [item for rec in recs for item in rec],
            'als_score': [item for rec in scores for item in rec],
            'als_rank': [rnk + 1 for rec in recs for rnk, _ in enumerate(rec)],
        })


@beartype
def train_als_model(
    data: Sessions, 
    n_items: int, 
    factors: int = 100, 
    iterations: int = 30,
    regularization: float = 0.01,
    alpha: Union[Float, Int] = 1.0,
    random_state: int = 1337,
    use_gpu: bool = True,
    num_threads: int = 0,
    calculate_training_loss: bool = False,
) -> Any:

    train_matrix = spm(data, n_items)

    recommender = AlternatingLeastSquares(
        factors=factors, 
        iterations=iterations, 
        random_state=random_state, 
        regularization=regularization, 
        alpha=alpha,
        use_gpu=use_gpu,
        num_threads=num_threads,
        calculate_training_loss=calculate_training_loss,
    )
    recommender.fit(train_matrix)

    return recommender
