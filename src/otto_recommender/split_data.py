import polars as pl
import tqdm
import numpy as np
import pandas as pd
from tqdm import tqdm

import gc
import implicit_features
import utils


df_test = pl.read_parquet('../data/test.par')
df_train = pl.read_parquet('../data/train.par')
df_train = df_train.with_column(
       pl.from_epoch("ts", unit="ms").alias('t_dat')
   )

df_test = df_test.with_column(
       pl.from_epoch("ts", unit="ms").alias('t_dat')
   )

type_labels = {'clicks': 0, 'carts': 1, 'orders': 2}

type_labels = pl.DataFrame({
    "type": list(type_labels.keys()),
    "values": list(type_labels.values())
})
df_train = df_train.join(type_labels, on="type", how="left").drop(columns=['type']).rename({'values': 'type'})
df_test = df_test.join(type_labels, on="type", how="left").drop(columns=['type']).rename({'values': 'type'})


df_test = df_test.rename({'session': 'customer_id', 'aid': 'article_id'})
df_train = df_train.rename({'session': 'customer_id', 'aid': 'article_id'})
test_users = df_test['customer_id'].unique()
df = pl.concat([df_train.select(df_test.columns), df_test])

df = df.sort('t_dat', reverse=True)
df = df.with_column(pl.col("ts").
               rank(method="ordinal", reverse=True).
               over(["customer_id"]).alias("num_action"))

# def split_data(df, validation_cut, test_users):
#     df_val = df.filter((pl.col("num_action") <= validation_cut) & (pl.col('customer_id').is_in(test_users)))
#     df_train = df.filter((pl.col("num_action") > validation_cut) | (~pl.col('customer_id').is_in(test_users)))
#     return df_train, df_val
def split_data(df, validation_cut, test_users):
    df_for_split = df.filter(pl.col('customer_id').is_in(test_users))
    df_for_split = df_for_split.join(df_for_split.groupby(['customer_id']).count(), on=['customer_id'])
    df_for_split_less = df_for_split.filter((pl.col('count')<=validation_cut)& (pl.col('count')>1))
    df_for_split_more = df_for_split.filter(pl.col('count')>validation_cut)
    test_less = df_for_split_less.filter(pl.col('num_action')==1)
    test_more = df_for_split_more.filter(pl.col("num_action") <= validation_cut)
    df_val = pl.concat([test_less, test_more])
    df = df.join(df_val, on=['customer_id', 'article_id', 'ts', 't_dat', 'type', 'num_action'], how='left')
    df_train = df.filter(pl.col('count').is_null())
    return df_train.drop(columns=['count']), df_val.drop(columns=['count'])

PREDICT_LB = True
if PREDICT_LB:
    print('PREDICT ON LB')
    df_train_all = df
    df_train_traks, df_test_traks = split_data(df_train_all, validation_cut=3, test_users=test_users)
    df_train_traks = df_train_traks.sort('t_dat', reverse=True)
    df_train_traks = df_train_traks.with_column(pl.col("ts").
                rank(method="ordinal", reverse=True).
                over(["customer_id"]).alias("num_action"))

    

del df_train
del df
     