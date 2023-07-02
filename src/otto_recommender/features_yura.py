def time_cat(df, days=7):
    return df.filter(pl.col('ts') >= df['ts'].max() - days*1000*60*60*24)

def make_features(df, df_traks, mode='metric'#, item_factors, user_factors
):
     # расстояние между действием и покупкой
     # Среднее время между событиями
     # фичи по времени и дню недели как профиль клиента 
     df_traks = df_traks.with_columns(pl.col('t_dat').dt.hour().alias('hour'))
     df_traks = df_traks.with_columns(pl.col('t_dat').dt.weekday().alias('weekday'))
     
     df_traks = df_traks.with_columns((pl.col('t_dat').dt.month()*100 + pl.col('t_dat').dt.day()).alias('day'))

     # Среднее время просмотра товара
     
     df_tmp = df_traks.filter(pl.col('ts_diff')<1000000).groupby(['article_id']).agg([pl.col('ts_diff').mean().alias('ts_diff_article_id_mean'), 
          pl.col('ts_diff').std().alias('ts_diff_article_id_std')])
     df = df.join(df_tmp, on=['article_id'], how='left')

     df_tmp = df_traks.filter(pl.col('ts_diff')<1000000).groupby(['customer_id']).agg([pl.col('ts_diff').mean().alias('ts_diff_customer_id_mean'), 
          pl.col('ts_diff').std().alias('ts_diff_customer_id_std')])
     df = df.join(df_tmp, on=['customer_id'], how='left')

     # Средняя неделя и час покупки товара
     df = df.join(df_traks.groupby(['customer_id']).agg(pl.col('day').max()), on=['customer_id'], how='left')
     df = df.join(df_traks.groupby(['article_id']).agg(pl.col('ts').min().alias('min_ts_article_id')), on=['article_id'], how='left')

     df = df.join(df_traks.groupby(['article_id', 'day']).count().rename({'count': 'count_item_in_last_day'}), 
          on=['article_id', 'day'], how='left')#.drop(columns=['day'])
     df_orders = df_traks.filter(pl.col('type')!=0)
     df = df.join(df_orders.groupby(['article_id', 'day']).count().rename({'count': 'count_item_in_last_day_buy'}), 
          on=['article_id', 'day'], how='left').drop(columns=['day'])
     # Сколько раз клиент смотрел товар
     df_count_in_session = df_traks.groupby(['article_id', 'customer_id']).count()
     df = df.join(df_count_in_session.groupby(['article_id']).agg(pl.col('count').mean().alias('repeat_action')), on=['article_id'], how='left')
     df_count_in_session = df_orders.groupby(['article_id', 'customer_id']).count()
     df = df.join(df_count_in_session.groupby(['article_id']).agg(pl.col('count').mean().alias('repeat_buy')), on=['article_id'], how='left')

     df_traks = df_traks.with_column(pl.col("ts").
                rank(method="ordinal", reverse=True).
                over(['customer_id', 'article_id']).alias("user_item_num"))

     # Как давно он покупал этот товар
     df_mean_click_to_buy = df_traks.filter(pl.col('type')==1)\
        .groupby(['customer_id','article_id']).agg(pl.col('user_item_num').min())\
        .groupby(['article_id']).agg(pl.col('user_item_num').mean().alias('mean_click_to_buy').cast(pl.Float32))
     df = df.join(df_mean_click_to_buy, on=['article_id'], how='left')
    
     df_rnn = get_rnn(mode)#.rename({'rnk': 'rnk_rnn'})
    # df_rnn = df_rnn.filter(pl.col('rnk')<=50)
     df = df.join(df_rnn, how='left', on=['customer_id', 'article_id'])

     for i in ['bm25', 'tfidf', 'cosine']:
        
        df_tmp = pl.read_parquet(f'../data/misha/parqs/cur_{i}_df_{misha_map[mode]}_preds_100.parquet')\
            .drop(columns=['index']).rename({'user': 'customer_id', 'item_id': 'article_id'})
      #  print(df_train.columns, df_tmp.columns)
        df = df.join(df_tmp, how='left', on=['customer_id', 'article_id'])
    
    # triplet = pl.read_parquet(f'data/rnn_cooc_3cooc/cur_cooc_triplets_{mode}.parquet').rename({'user': 'customer_id', 'item_id': 'article_id'})
    # triplet = triplet.filter(pl.col('cooc_triplet_rank_0')<=50)
    # df_train = df_train.join(triplet, how='outer', on=['customer_id', 'article_id'])

     coo = pl.read_parquet(f'../data/misha/parqs/cur_cooc_smart_df_{misha_map[mode]}_preds_100.parquet').rename({'user': 'customer_id', 'item_id': 'article_id'})
    # coo = coo.filter(pl.col('cooc_rank_0')<=30).drop(columns=['index'])
     df = df.join(coo, how='left', on=['customer_id', 'article_id'])
     rnk_cols = ['bm25_rank', 'tfidf_rank', 'cosine_rank', 'cooc_rank_0', 'rnn_rnk']
     df = df.with_columns(df[rnk_cols].min(axis=1).alias('min_rnk_cand_misha').cast(pl.Float32))
        
     last_view = df_traks.groupby(['article_id']).agg(pl.col('ts').max() ).with_columns(((df_traks['ts'].max() - pl.col('ts'))//1000).alias('last_view'))
     df = df.join(last_view.select(['article_id', 'last_view']), how='left', on=['article_id'])

     
     last_view = df_orders.groupby(['article_id']).agg(pl.col('ts').max() ).with_columns(((df_traks['ts'].max() - pl.col('ts'))//1000).alias('last_buy'))
     df = df.join(last_view.select(['article_id', 'last_buy']), how='left', on=['article_id'])
     joint_traks = pl.read_parquet(f'../data/{mode}_coo_matrix.par')
    # joint_traks = get_joint_listren(df_traks, 'article_id')
     df = agg_ratio_joint_sales(df, joint_traks, df_traks, pred_users=df['customer_id'].unique(), N=30)
     df = agg_ratio_joint_sales(df, joint_traks, df_traks, pred_users=df['customer_id'].unique(), N=15)
     #return df, joint_traks
     df = agg_ratio_joint_sales(df, joint_traks, df_traks, pred_users=df['customer_id'].unique(), N=5)
     df = agg_ratio_joint_sales(df, joint_traks, df_traks, pred_users=df['customer_id'].unique(), N=1)
     del joint_traks
   # return df
     df_orders = df_orders.with_column(pl.col("ts").
          rank(method="min", reverse=True).
          over(["customer_id"]).alias("num_action"))
     joint_traks = create_cooc_stats(df_orders, min_rank_diff = -3, max_rank_diff = 3, items_per_item=3)

     df = agg_ratio_joint_sales(df, joint_traks, df_orders, pred_users=df['customer_id'].unique(), N=2)
     df = agg_ratio_joint_sales(df, joint_traks, df_orders, pred_users=df['customer_id'].unique(), N=10)

     df_tmp = df_traks.groupby(['customer_id', 'article_id']).agg([pl.col('num_action').min().alias('min_num_action'), 
          pl.col('num_action').count().alias('user_item_count')])
     df = df.join(df_tmp, on=['customer_id', 'article_id'], how='left')
     #return df
     df_tmp = df_orders.groupby(['customer_id', 'article_id']).agg(pl.col('num_action').count().alias('user_item_count_orders'))
     df = df.join(df_tmp, on=['customer_id', 'article_id'], how='left')
    
     df_tmp = df_traks.groupby(['customer_id', 'article_id']).agg(pl.col('num_action').count().alias('user_item_count_all'))
     df = df.join(df_tmp, on=['customer_id', 'article_id'], how='left')
    
     df = df.with_columns((pl.col(f'user_item_count_all') / pl.col(f'mean_click_to_buy')).alias('ratio_num_click_to_buy').cast(pl.Float32))
    
     # df = df.with_columns((pl.col('user_item_count_orders')* pl.col('repeat_buy')).alias('prob_buy_twice'))
     # df = df.with_columns((pl.col('user_item_count')* pl.col('repeat_action')).alias('prob_action_twice'))

     # df = pd.merge(df, joint_traks.rename({'article_id_x': 'article_id', 'article_id_y': 'last_article_id'}, axis=1), how='left', on=['article_id', 'last_article_id'])
     # df = df.drop(columns=['last_article_id'])
     df_implicit = pl.read_parquet(f'../data/implicit_lb_{mode}.par')#.drop(columns=['__index_level_0__']) # predict
     df = df.join(df_implicit, on=['article_id', 'customer_id'], how='left')
     df_implicit = pl.read_parquet(f'../data/implicit_lb_{mode}10.par')#.drop(columns=['__index_level_0__']) # predict
     df = df.join(df_implicit, on=['article_id', 'customer_id'], how='left')

     #return df
     # df_implicit = pd.read_parquet(f'data/implicit_lb_{mode}_orders.par') # predict
     # df = pd.merge(df, df_implicit, on=['article_id', 'customer_id'], how='left')

     gc.collect()


     # Среднии, минимальные и максимальные ранги после модели первого уровня для артистов и треков
     rnk_cols = ['rnk_tifuknn_tfidf', 'rnk_tifu_cos', 'rnk_tfidf', 'rnk_cos412', 'rnk_BM25']
     df = df.with_columns(df[rnk_cols].mean(axis=1).alias('mean_rnk_cand').cast(pl.Float32))
     df = df.with_columns(pl.Series(df.select(rnk_cols).to_pandas().std(axis=1)).alias('std_rnk_cand'))
     df = df.with_columns(df[rnk_cols].min(axis=1).alias('min_rnk_cand').cast(pl.Float32))
     df = df.with_columns(df[rnk_cols].max(axis=1).alias('max_rnk_cand').cast(pl.Float32))

     rnk_cols_10 = ['rnk_tifuknn_tfidf_10', 'rnk_tifu_cos_10', 'rnk_tfidf_10', 'rnk_cos412_10', 'rnk_BM25_10']
     df = df.with_columns(df[rnk_cols_10].mean(axis=1).alias('mean_rnk_cand_10').cast(pl.Float32))
     df = df.with_columns(pl.Series(df.select(rnk_cols_10).to_pandas().std(axis=1)).alias('std_rnk_cand_10'))
     df = df.with_columns(df[rnk_cols_10].min(axis=1).alias('min_rnk_cand_10').cast(pl.Float32))
     df = df.with_columns(df[rnk_cols_10].max(axis=1).alias('max_rnk_cand_10').cast(pl.Float32))

     all_rnk_cols = rnk_cols_10 + rnk_cols
     df = df.with_columns(df[all_rnk_cols].mean(axis=1).alias('mean_rnk_cand_all').cast(pl.Float32))
     df = df.with_columns(pl.Series(df.select(all_rnk_cols).to_pandas().std(axis=1)).alias('std_rnk_cand_all'))

     df = df.with_column(pl.col("mean_rnk_cand_all").
          rank(method="dense", reverse=True).
          over(["customer_id"]).alias("rnk_all_cand"))

     df = df.with_columns(df.select([ pl.col(col).is_not_null().cast(pl.Int16) for col in all_rnk_cols]).sum(axis=1).alias('num_cand'))
     # фичи на популярность пользователя, товара 
     tmp = df_traks.groupby(['article_id']).count().rename({'count':'pop_article_id'})
     df = df.join(tmp, on=['article_id'], how='left')
     # Ранги песен внутри кандидатов
     df = df.with_column(pl.col("pop_article_id").
          rank(method="dense", reverse=True).
          over(["customer_id"]).alias("pop_item_rnk_user"))

     df_traks = df_traks.join(tmp, on=['article_id'], how='left')
     df = df.join(df_traks.groupby(['customer_id']).agg(pl.col('article_id').n_unique().alias('nunique_article_id')), on=['customer_id'], how='left')
     
     tmp = df_traks.groupby(['customer_id']).count().rename({'count':'pop_user'})
     df = df.join(tmp, on=['customer_id'], how='left')
     tmp = df_traks.filter(pl.col('type')==1).groupby(['customer_id']).count().rename({'count':'pop_user_carts'})
     df = df.join(tmp, on=['customer_id'], how='left')
     tmp = df_traks.filter(pl.col('type')==2).groupby(['customer_id']).count().rename({'count':'pop_user_orders'})
     df = df.join(tmp, on=['customer_id'], how='left')
     df = df.with_columns((pl.col(f'pop_user') / pl.col(f'pop_user_orders')).alias('cr_to_buy_user').cast(pl.Float32))
     gc.collect()
     
     # Какие треки в среднем по популярности слушает пользователь
     tmp = df_traks.groupby(['customer_id']).agg([pl.col('pop_article_id').std().alias('pop_article_id_std_user'), 
          pl.col('pop_article_id').mean().alias('pop_article_id_mean_user')])
     df = df.join(tmp, on=['customer_id'], how='left')
     df = df.with_columns((pl.col('pop_article_id_mean_user')/ pl.col('pop_article_id')).alias('ratio_pop_user_item').cast(pl.Float32))
     #return df
     for i in [2, 5, 10, 15]:
          tmp_track = time_cat(df_traks, days=i)
         # tmp_track = df_traks[df_traks['num_action']<=i]

          tmp = tmp_track.groupby(['article_id']).count().rename({'count':f'pop_item_{i}'})
          df = df.join(tmp, on=['article_id'], how='left')
          df = df.with_column(pl.col(f'pop_item_{i}').
               rank(method="dense", reverse=True).
               over(["customer_id"]).alias(f"pop_item_rnk_user_{i}"))

          tmp = tmp_track.filter(pl.col('type')==2).groupby(['article_id']).count().rename({'count': f'pop_item_orders_{i}'})
          df = df.join(tmp, on=['article_id'], how='left')
          gc.collect()
          tmp = tmp_track.filter(pl.col('type')==1).groupby(['article_id']).count().rename({'count': f'pop_item_cart_{i}'})
          df = df.join(tmp, on=['article_id'], how='left')
          gc.collect()
          df = df.with_columns((pl.col(f'pop_item_orders_{i}') / pl.col(f'pop_item_{i}')).alias(f'cr_to_buy_item_{i}').cast(pl.Float32))
          # Добавить события пользователя
     df = df.with_columns((pl.col(f'pop_item_10') / pl.col(f'count_item_in_last_day')).alias(f'mean_pop_on_pop_in_last_day').cast(pl.Float32))
     
     gc.collect()
     #return df
     print('start_time_features')
     tmp = df_traks.groupby('customer_id').agg([pl.col('ts').min().alias('min_date_session'), 
           pl.col('ts').max().alias('max_date_session')])
     tmp = tmp.with_columns(((pl.col('max_date_session') - pl.col('min_date_session'))//1000).alias('leng_session'))
     df = df.join(tmp.select(['customer_id', 'leng_session']), how='left', on=['customer_id'])
     df = df.with_columns((pl.col('pop_user') / pl.col('leng_session')).alias('items_in_seconds').cast(pl.Float32))

     tmp = df_traks.groupby(['article_id']).agg([
          #pl.col('weekday').median().alias('median_weekday_article_id').cast(pl.Float32),
          #pl.col('weekday').mean().alias('mean_weekday_article_id').cast(pl.Float32),
          pl.col('hour').median().alias('median_hour_article_id').cast(pl.Float32),
          pl.col('hour').mean().alias('mean_hour_article_id').cast(pl.Float32),
        pl.col('type').mean().alias('mean_type_article_id').cast(pl.Float32),]
     )
     df = df.join(tmp, on=['article_id'], how='left')

     tmp = df_traks.groupby(['customer_id']).agg([
          #pl.col('weekday').median().alias('median_weekday_customer_id').cast(pl.Float32),
          #pl.col('weekday').mean().alias('mean_weekday_customer_id').cast(pl.Float32),
          pl.col('hour').median().alias('median_hour_customer_id').cast(pl.Float32),
          pl.col('hour').mean().alias('mean_hour_customer_id').cast(pl.Float32),
         pl.col('type').mean().alias('mean_type_customer_id').cast(pl.Float32),]
     )
     df = df.join(tmp, on=['customer_id'], how='left')

     df = df.with_columns((pl.col('mean_hour_article_id') / pl.col('mean_hour_customer_id')).alias('mean_hour_diff').cast(pl.Float32))
        
     gc.collect()
     rnn_emb = pl.read_parquet('../data/rnn_15.par')
     df = df.join(rnn_emb, on=['article_id'], how='left')
     #df = df[(df['rnk_tifuknn_tfidf']>=0) | (df['rnk_tifu_cos']>=0)]
     return df