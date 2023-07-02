import argparse

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import gc

import lightgbm as lgb

from utils import truncate_list


parser = argparse.ArgumentParser()
# parser.add_argument('--candidates', type=str, default='../../data/07_model_output/train.parquet')
# parser.add_argument('--boosters', nargs='+', type=str)
parser.add_argument('--out', type=str, default='../../data/07_model_output/preds_try.csv')
parser.add_argument('--col_user', type=str, default='session')


if __name__ == '__main__':
    args = parser.parse_args()
    path_boosters = [
        "booster_fold0.lgb", "booster_fold1.lgb", "booster_fold2.lgb"
    ]
    boosters = [
        lgb.Booster(model_file=booster)
        for booster in path_boosters
    ]

    sub = pd.read_csv("../../data/01_raw/sample_submission.csv")

    PATH_CHUNKS = '../../data/07_model_output/test_w_features_'
    dfs = list()

    for chunk_id in tqdm(range(17), desc='predicting'):
        chunk_name = f'{PATH_CHUNKS}{chunk_id}.parquet'
        test_candidates = pl.read_parquet(chunk_name).to_pandas()
        df = test_candidates[[args.col_user, 'item_id']]
        vals = test_candidates[boosters[0].feature_name()].to_numpy(dtype=np.float32)
        del test_candidates

        preds = np.zeros(len(df))
        for model in boosters:
            preds += model.predict(vals, num_threads=20)

        df['preds'] = preds
        df = df.sort_values([args.col_user, 'preds'], ascending=[True, False])
        df[args.col_user] = list(map(str, df[args.col_user]))
        df = pl.from_pandas(df)
        df = (
            (
                df.groupby([args.col_user]).agg(
                    [
                        pl.col("item_id").list(),
                    ]
                )
            )
            .sort(args.col_user)
            .to_pandas()
        )
        df['item_id'] = df['item_id'].apply(lambda x: truncate_list(list(x), max_len=20, from_start=True))
        list_labels = df['item_id'].apply(lambda x: ' '.join(str(aid) for aid in x))

        for session_type in ['clicks', 'carts', 'orders']:
            df_cur = df[[args.col_user, "item_id"]].copy()
            df_cur.rename({"item_id": 'labels', args.col_user: 'session_type'}, axis=1, inplace=True)
            df_cur['session_type'] = df_cur['session_type'].apply(lambda x: x + '_' + session_type)
            df_cur['labels'] = list_labels
            dfs.append(pl.from_pandas(df_cur))

        del df_cur, df, vals
        gc.collect()

    dfs = pl.concat(dfs)
    dfs = pl.from_pandas(sub[['session_type']]).join(dfs, on=['session_type'])
    dfs.write_csv(args.out)
