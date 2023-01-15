import numpy as np
import pandas as pd
import implicit
import scipy.sparse as sparse
from sklearn import preprocessing

from tqdm import tqdm

tqdm.pandas()
import warnings
warnings.filterwarnings('ignore')

def tifuknn_score(df_action, m, r):
    """
    Экспоненциальное сглаживание. Чем дальше от последнего взаимодействия, тем меньше его скор
    :param m: количество разбиений
    :param r: коэффициент затухания
    """
    df_action['rnk'] = df_action.groupby('customer_id')['t_dat'].rank(method='dense', ascending=False)-1
    df_action['max_days_view'] = df_action['customer_id'].map(df_action.groupby('customer_id')['t_dat'].nunique()).astype(int)
    df_action['el_in_batch'] = df_action['max_days_view']//m+1
    df_action['num_batch'] = df_action['rnk']//df_action['el_in_batch']
    df_action['tifuknn_score'] = r**(df_action['num_batch'])
    df_action = df_action.drop(columns=['max_days_view', 'el_in_batch', 'num_batch', 'rnk'])
    return df_action

def get_predicion(model, data_conf, name_model='', nonactual_items=[], pred_users=[], filter_already_liked_items=True):
    df_pred = []
    X_train = data_conf.tocsr()
    for i in tqdm(pred_users):
        raw_recs  = model.recommend(userid=i,
                                   user_items=X_train[i],
                                   N=100,
                                   filter_already_liked_items=filter_already_liked_items,
                                   recalculate_user=True,
                                  # filter_items=nonactual_items
                                   )
        raw_recs = np.vstack(raw_recs).T
        df_pred.append(np.c_[raw_recs, len(raw_recs)*[i]])
        #df_pred.append(add_columns(i, raw_recs))
    df_pred = pd.DataFrame(np.vstack(df_pred), columns = ['article_id', 'score_'+name_model, 'customer_id'] )
    df_pred = df_pred[df_pred['score_'+name_model]>0]
    #df_pred['rnk_'+name_model] = df_pred.groupby(['customer_id'])['score_'+name_model].rank(method='first', ascending=False).astype(int)
    if len(nonactual_items)>0:
        df_pred = filter_nonactual_items(df_pred, nonactual_items, name_model, rnk=50 )

   # df_pred = df_pred[~df_pred['article_id'].isin(nonactual_items)]
    df_pred['rnk_'+name_model] = df_pred.groupby(['customer_id'])['score_'+name_model].rank(method='first', ascending=False).astype(int)
    df_pred = df_pred[df_pred['rnk_'+name_model]<=50]
    #df_pred['rnk_'+name_model] = df_pred.groupby(['customer_id'])['score_'+name_model].rank(method='first', ascending=False)
    return df_pred

def filter_nonactual_items(df, nonactual_items, name_model, rnk=50, ):
    nonactual_items = pd.DataFrame(nonactual_items,columns=['article_id'])
    nonactual_items['nonactual_items'] = df['score_'+name_model].max()+1
    df = pd.merge(df, nonactual_items, on=['article_id'], how='left')
    df['nonactual_items'] = df['article_id'].map(nonactual_items.set_index('article_id')['nonactual_items'])
    df['nonactual_items'] = df['nonactual_items'].fillna(0)
    df['rnk_new'] = df['score_'+name_model] + df['nonactual_items']
    df['rnk'] = df.groupby(['customer_id'])['rnk_new'].rank(method='first', ascending=False)
    df = df[df['rnk']<=rnk]
    df = df.drop(columns=['nonactual_items', 'rnk', 'rnk_new'])
    return df

    
def add_columns(value, raw_recs):
    added_list = []
    for i in raw_recs:
        k = list(i)
        k.append(value)
        added_list.append(k)
    return added_list

def split_data(df, validation_days=7):
    """ Split a pandas dataframe into training and validation data, using <<validation_days>>
    """
    df_train = df[df['num_action'] > validation_days]
    df_val = df[df['num_action'] <= validation_days]
    return df_train, df_val
    
def make_pred(df, pred_users, train_day=1000, filter_already_liked_items=True):
   # df = tifuknn_score(df, 17, 0.2996714759625573)
    _, df = split_data(df, validation_days=train_day)
    df = tifuknn_score(df, 236.52, 0.911)
   # df = tifuknn_score(df, 20, 0)
    item_count = df['article_id'].value_counts()
    valid_item = item_count[item_count>5].keys()
    df = df[df['article_id'].isin(valid_item)]

    le_item = preprocessing.LabelEncoder()
    le_user = preprocessing.LabelEncoder()
    df['customer_id_le'] = le_user.fit_transform(df['customer_id']).astype(np.int32)
    df['article_id_le'] = le_item.fit_transform(df['article_id']).astype(np.int32)

    pred_users = le_user.transform(list(set(pred_users).intersection(set(df['customer_id']))))

    df['rating'] =  df['tifuknn_score'].astype(np.float32)
    df_train = sparse.coo_matrix((list(df.rating + 0.01),
                                    (list(df.customer_id_le), list(df.article_id_le))))
    model = implicit.nearest_neighbours.TFIDFRecommender(K=427)
    model.fit(df_train.tocsr().astype(np.float32))
    last_buy = df.groupby('article_id_le')['t_dat'].max()
    nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=20))].keys())
   # nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=2))].keys())
    df_pred = get_predicion(model, df_train, 'tifuknn_tfidf', nonactual_items, pred_users, filter_already_liked_items)
    #df_pred = filter_non_actual(df_pred, actual_items, 'tifuknn_tfidf')

    df = tifuknn_score(df, 7, 0.6495145290244319)
    df['rating'] = df['tifuknn_score'].astype(np.float32)
    df_train = sparse.coo_matrix((list(df.rating + 0.01),
                                (list(df.customer_id_le), list(df.article_id_le))))
    #nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=1))].keys())
#     model = implicit.nearest_neighbours.CosineRecommender(K=100)
#     model.fit(df_train.T.tocsr().astype(np.float32))
#     df_pred_cos = get_predicion(model, df_train, 'tifuknn_cos', nonactual_items, pred_users)
#    # df_pred_cos = filter_non_actual(df_pred_cos, actual_items, 'tifuknn_cos')
#     df_pred = pd.merge(df_pred, df_pred_cos, on=['customer_id', 'article_id'], how='outer')
#     del df_pred_cos
   
   # nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=20))].keys())
    model = implicit.nearest_neighbours.CosineRecommender(K=50)
    model.fit(df_train.tocsr().astype(np.float32))
    df_pred_cos = get_predicion(model, df_train, 'tifu_cos', nonactual_items, pred_users, filter_already_liked_items)
   # df_pred_cos = filter_non_actual(df_pred_cos, actual_items, 'cos')
    df_pred = pd.merge(df_pred, df_pred_cos, on=['customer_id', 'article_id'], how='outer')
    del df_pred_cos

    df['rating'] = 1
   # nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=20))].keys())
    df_train = sparse.coo_matrix((list(df.rating + 0.01),
                                    (list(df.customer_id_le), list(df.article_id_le))))
    nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=118))].keys())
    #nonactual_items = []
    model = implicit.nearest_neighbours.TFIDFRecommender(K=495)
    model.fit(df_train.tocsr().astype(np.float32))
    df_pred_tfidf = get_predicion(model, df_train, 'tfidf', nonactual_items, pred_users, filter_already_liked_items)
   # df_pred_tfidf = filter_non_actual(df_pred_tfidf, actual_items, 'tfidf')
    df_pred = pd.merge(df_pred, df_pred_tfidf, on=['customer_id', 'article_id'], how='outer')
    del df_pred_tfidf

  #   model = implicit.nearest_neighbours.TFIDFRecommender(K=1)
  #   model.fit(df_train.T.tocsr().astype(np.float32))
  #   df_pred_tfidf = get_predicion(model, df_train, 'tfidf1', nonactual_items, pred_users)
  #  # df_pred_tfidf = filter_non_actual(df_pred_tfidf, actual_items, 'tfidf')
  #   df_pred = pd.merge(df_pred, df_pred_tfidf, on=['customer_id', 'article_id'], how='outer')
  #   del df_pred_tfidf

    model = implicit.nearest_neighbours.CosineRecommender(K=491)
    model.fit(df_train.tocsr().astype(np.float32))
    df_pred_tfidf = get_predicion(model, df_train, 'cos412', nonactual_items, pred_users, filter_already_liked_items)
   # df_pred_tfidf = filter_non_actual(df_pred_tfidf, actual_items, 'tfidf')
    df_pred = pd.merge(df_pred, df_pred_tfidf, on=['customer_id', 'article_id'], how='outer')
    del df_pred_tfidf

    #nonactual_items = list(last_buy[last_buy<(df['t_dat'].max() - pd.Timedelta(days=1))].keys())
    model = implicit.nearest_neighbours.BM25Recommender(K=361, B=0.46400215487774105, K1=1.1668)
    model.fit(df_train.tocsr().astype(np.float32))
    df_pred_bm = get_predicion(model, df_train, 'BM25', nonactual_items, pred_users, filter_already_liked_items)
   # df_pred_tfidf = filter_non_actual(df_pred_tfidf, actual_items, 'tfidf')
    df_pred = pd.merge(df_pred, df_pred_bm, on=['customer_id', 'article_id'], how='outer')
    del df_pred_bm

    df_pred['customer_id'] = le_user.inverse_transform(df_pred['customer_id'].astype(int))
    df_pred['article_id'] = le_item.inverse_transform(df_pred['article_id'].astype(np.int32))
    #df_pred['article_id'] = df_pred['article_id'].astype(str)
    # filter actual

   # df_pred = df_pred[df_pred['article_id'].isin(actual_items)]
    for col in set(df_pred.columns) - set(['article_id', 'customer_id']):
        df_pred[col] = df_pred[col].astype(np.float32)

    return df_pred

def tifuknn_score(df_action, m, r):
    """
    Экспоненциальное сглаживание. Чем дальше от последнего взаимодействия, тем меньше его скор
    :param m: количество разбиений
    :param r: коэффициент затухания
    """
    df_action['rnk'] = df_action.groupby('customer_id')['t_dat'].rank(method='dense', ascending=False)-1
    df_action['max_days_view'] = df_action['customer_id'].map(df_action.groupby('customer_id')['t_dat'].nunique())
    df_action['el_in_batch'] = df_action['max_days_view']//m+1
    df_action['num_batch'] = df_action['rnk']//df_action['el_in_batch']
    df_action['tifuknn_score'] = r**(df_action['num_batch'])
    return df_action

def filter_non_actual(df_pred, actual_items, name_model):
    df_pred = df_pred[df_pred['article_id'].isin(actual_items)]
    df_pred['rnk_'+name_model] = df_pred.groupby(['customer_id'])['score_'+name_model].rank(method='first', ascending=False)
    df_pred = df_pred[df_pred['rnk_'+name_model]<=50]
    return df_pred


def als_predict(model, actual_items):
    actual_items = np.array(actual_items)
    pred_als = []
    for item in tqdm(range(len(model.user_factors))):
        pred_tmp = model.user_factors[item].dot(model.item_factors.T)
        pred_tmp = pred_tmp[actual_items]
        top_inx = np.argpartition(pred_tmp, -100)[-100:]
        df_tmp = np.vstack([pred_tmp[top_inx], actual_items[top_inx], 100*[item]])
        pred_als.append(df_tmp)
    pred_als = pd.DataFrame(np.hstack(pred_als).T, columns=['score_als', 'article_id', 'customer_id'])
    return pred_als


def als(df, pred_users, name_model='als'):
    df = tifuknn_score(df, 2, 0.3021)
    item_count = df['article_id'].value_counts()
    valid_item = item_count[item_count>1].keys()
    df = df[df['article_id'].isin(valid_item)]

    le_item = preprocessing.LabelEncoder()
    le_user = preprocessing.LabelEncoder()
    df['customer_id_le'] = le_user.fit_transform(df['customer_id'])
    df['article_id_le'] = le_item.fit_transform(df['article_id'])
    #df['rating'] = 1
    df['rating'] = df['tifuknn_score'].astype(np.float32)
    pred_users = le_user.transform(list(set(pred_users).intersection(set(df['customer_id']))))
    df_train = sparse.coo_matrix((list(df.rating + 0.01),
                                    (list(df.customer_id_le), list(df.article_id_le))))

    model = implicit.als.AlternatingLeastSquares(factors=100, iterations=30)
    model.fit(df_train.tocsr().astype(np.float32))
    X_train = df_train.tocsr()
   # pred_users = le_user.transform(pred_users)
    
    #pred_users = list(range(X_train.shape[0]))
    pred = model.recommend(pred_users, X_train[pred_users], filter_already_liked_items=False, N=50)
    i = 0
    result = []
    for items, score in zip(pred[0], pred[1]):
        pred_tmp = np.vstack([items, score]).T
        pred_tmp = add_columns(pred_users[i], pred_tmp)
        result.append(pred_tmp)
        i += 1
    result = np.vstack(result)
    df_pred = pd.DataFrame(result, columns=['article_id', 'score_'+name_model, 'customer_id'])
    df_pred['customer_id'] = le_user.inverse_transform(df_pred['customer_id'].astype(int))
    df_pred['article_id'] = le_item.inverse_transform(df_pred['article_id'].astype(int))
    return df_pred


def make_score_implicit_product(df, pred_users, col_item='product_type_no'):
    item_count = df[col_item].value_counts()
    valid_item = item_count[item_count>1].keys()
    df = df[df[col_item].isin(valid_item)].copy()

    le_item = preprocessing.LabelEncoder()
    le_user = preprocessing.LabelEncoder()
    df['customer_id_le'] = le_user.fit_transform(df['customer_id'])
    df['article_id_le'] = le_item.fit_transform(df[col_item])
    
    pred_users = le_user.transform(list(set(pred_users).intersection(set(df['customer_id']))))

    df['rating'] = 1
    df_train = sparse.coo_matrix((list(df.rating + 0.01),
                                    (list(df.customer_id_le), list(df.article_id_le))))
    model = implicit.nearest_neighbours.TFIDFRecommender(K=200)
    model.fit(df_train.T.tocsr().astype(np.float32), show_progress=False)
    df_pred = get_predicion(model, df_train, col_item, pred_users=pred_users)
    df_pred = df_pred.rename({'article_id': col_item}, axis=1)
    return df_pred