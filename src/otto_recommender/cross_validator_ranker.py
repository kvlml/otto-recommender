import os

import pandas as pd
import numpy as np

from functools import wraps
import time

import gc
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold
from pandas import DataFrame
from typing import List
import time

from lightgbm import LGBMRanker
import catboost as cbs
import xgboost as xgb

from sklearn.metrics import classification_report, roc_auc_score, log_loss

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


class CrossValidator:

    def __init__(self,
            df_train,
            target_col,
            idx_col=None,
            feat_to_try=None,
            best_params=None,
            random_state=42,
            kf=None,
            n_splits=5,
            plot=False,
            early_stop_rounds=50,
            model_type='lgbm',
            model=None,
            metric=roc_auc_score,
        ):

        self.target_col = target_col
        self.idx_col = idx_col

        self.feat_to_try = feat_to_try

        self.cols_to_test = [self.idx_col, self.target_col] if self.idx_col is not None else [self.target_col]
        if self.feat_to_try is None:
            self.feat_to_try = list(df_train.drop(self.cols_to_test, axis=1).columns)

        self.X = df_train[self.feat_to_try].reset_index(drop=True)
        self.y = df_train[self.target_col].reset_index(drop=True)

        self.to_check = df_train[self.cols_to_test].copy().reset_index(drop=True)
        self.to_check['valid_fold_num'] = np.zeros(self.to_check.shape[0])
        self.to_check['prediction'] = np.zeros(self.to_check.shape[0])

        self.best_params = best_params
        self.plot = plot
        self.kf = kf
        self.n_splits = n_splits
        self.random_state = random_state
        self.early_stop_rounds = early_stop_rounds

        self.model_type = model_type
        self.models = list()
        self.model = model
        self.metric = metric


    @property
    def is_fitted(self):
        if len(self.models) < self.n_splits:
            print('Models are not fitted')
            return False
        else:
            return True


    def do_prediction(self, model, data):
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(data)[:,1]
        elif hasattr(model, 'predict'):
            return model.predict(data)


    def do_cv(self):

        if kf is None:
            kf = GroupKFold(n_splits=self.n_splits, shuffle=False, random_state=self.random_state)

        if self.metric is None:
            self.metric = roc_auc_score

        if self.best_params is None:
            self.best_params = dict()
        for param in ['num_leaves', 'min_data_in_leaf', 'max_depth', 'n_estimators']:
            if param in self.best_params:
                self.best_params[param] = int(self.best_params[param])

        params = dict()

        if self.model_type == 'lgbm':
            params = {
            #    'objective': 'binary',
            #          'boosting_type': 'gbrt',
                      'boosting_type': 'goss',
                      'learning_rate': 0.04,
                      'n_estimators': 10000,                
                      'metric': 'auc',
                      'seed': self.random_state,
                      'max_depth': -1,
                      'verbose': False,
                      'n_jobs': 40,
            #          'is_unbalance': True,
            #          'boost_from_average': False,
            #          'feature_fraction_seed': 1337,
            #        'bagging_seed': 1337,
            #        'drop_seed': 1337,
            #        'data_random_seed': 1337,
                     }

        elif self.model_type == 'xgb':
            params = {
                    'n_estimators': 10000,
                    'max_depth': 8,
                    'learning_rate': 0.04,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'random_state': self.random_state,
                    'eval_metric': 'auc',
                    # 'tree_method': 'gpu_hist'
                     }

        elif self.model_type == 'cbs':
            params = {
                  # 'bagging_temperature': 0,
                    #   boosting_type='Plain',
                    #   'bootstrap_type': 'Bernoulli',
                    #  'subsample': 0.9,
                   #     'rsm': 0.75,
                        'max_depth': 8,
                    #    loss_function='CrossEntropy',
                   # loss_function = FocalLossObjective(),
                    #   scale_pos_weight=3.7, #1.15
                       'random_strength': 1,
                       'thread_count': 38,
                     #   'task_type': "GPU",
                       'iterations': 10000,
                       'l2_leaf_reg': 3, 
                       # reg_lambda = 10,
                       'learning_rate': 0.04,
                       'save_snapshot': False,
                       'snapshot_file': 'snapshot_best.bkp',
                       'random_state': self.random_state, #38
                       'od_type': 'Iter',
                       'od_wait': 100,
                       'custom_loss': ['AUC'],
                        'max_ctr_complexity': 3,
                       'eval_metric': 'AUC',
                        'verbose': False,
                       'use_best_model': True
                     }
        params.update(self.best_params)

        metrics = list()

        print('here we go again')
        print()

        for i, (train_index, valid_index) in enumerate(kf.split(self.X, list(self.y))):

            X_train = self.X.loc[train_index]
            X_valid = self.X.loc[valid_index]

            y_train = self.y.loc[train_index]
            y_valid = self.y.loc[valid_index]

            t = time.time()

            if self.model_type == 'xgb':
                model = xgb.XGBRanker(**params)
               # print('model is defined')
                model.fit(X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc',
                    early_stopping_rounds=self.early_stop_rounds,
                        verbose=50)

            elif self.model_type == 'lgbm':
                model = LGBMClassifier(**params)
               # print('model is defined')
                model.fit(X_train, y_train, 
                          eval_set=[(X_valid, y_valid)],
                          eval_metric='auc',                          
                         verbose=50, 
                        early_stopping_rounds=self.early_stop_rounds
                       )

            elif self.model_type == 'cbs':              
                model = cbs.CatBoostRanker(
                     #   bagging_temperature=0,
                    #   boosting_type='Plain',
                       bootstrap_type='Bernoulli',
                      subsample=0.9,
                        rsm=0.75,
                        max_depth=7,
                    #    loss_function='CrossEntropy',
                   # loss_function = FocalLossObjective(),
                    #   scale_pos_weight=3.7, #1.15
                       random_strength=1,
                       thread_count=38,
                      #  task_type = "GPU",
                       iterations=10000,
                       langevin=True,
                       l2_leaf_reg = 3, 
                       # reg_lambda = 10,
                       learning_rate=0.02,
                       save_snapshot=False,
                       snapshot_file='snapshot_best.bkp',
                       random_state=43, #38
                       od_type='Iter',
                       od_wait=50,
                       #custom_loss=['AUC'],
                        max_ctr_complexity=3,
                       eval_metric='AUC',
                        verbose=False,
                       use_best_model=True
                        )
               # print('model is defined')
                model.fit(
                    X_train, y_train,
                    #   cat_features=new_cat_features,
                    #   verbose=False,
                    eval_set=(X_valid, y_valid),
                    #   logging_level='Silent',
                    plot=self.plot
                )
            else:
                model = self.model
                model.fit(X_train, y_train)

            print()
            print('fit duration:', round((time.time() - t) / 60, 3), 'minutes')

            self.models.append(model)

            if self.plot:
                feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, self.X.columns)), columns=['Value','Feature'])\
                    .sort_values(by="Value", ascending=False).head(50)

                plt.figure(figsize=(20, 10))
                sns.barplot(x="Value", y="Feature", data=feature_imp)
                plt.title('Features (cur fold)')
                plt.tight_layout()
                plt.show();

            y_pred = self.do_prediction(model, X_valid)
            print('valid predict is done')
            metric_cur = metric(y_valid, y_pred)
            metrics.append(metric_cur)

            self.to_check.loc[valid_index, 'prediction'] = y_pred
            self.to_check.loc[valid_index, 'valid_fold_num'] = i

            print(i + 1, "metric:", round(metric_cur, 5))
            print('_' * 50)
            print()

        print("avg metric:", np.mean(metrics))
        print("avg metric std:", np.std(metrics))
        print("OOF metric:", metric(self.y.values, self.to_check['prediction'].values))
        print()

        return self.to_check


    def predict(self, df_test):
        if not self.is_fitted:
            return

        pred = np.zeros(df_test.shape[0])
        for model in self.models:
            prediction_skf = self.do_prediction(model, df_test[self.feat_to_try])
            pred += prediction_skf / len(self.models)

        df_pred = df_test[[self.idx_col]].copy().reset_index(drop=True) if self.idx_col is not None else pd.DataFrame()
        df_pred[self.target_col] = pred

        return df_pred


    @property
    def _feature_importance(self, importance_type="gain"):
        if not self.is_fitted:
            return

        importances = np.zeros(len(self.feat_to_try))

        for model in self.models:
            importances += model.feature_importances_ / len(self.models)
   
        feature_imp = pd.DataFrame(sorted(zip(importances, self.X.columns)), columns=['Value', 'Feature'])\
                    .sort_values(by="Value", ascending=False).reset_index(drop=True)

        return feature_imp


    def plot_feature_importance(self, n_top_feats=None): 

        if n_top_feats is None:
            n_top_feats = len(self.feat_to_try)

        title = "Importance"
        x_label = "Importance"
        y_label = "Features"

        feature_imp = self._feature_importance

        tuples = [
            (name, score) for name, score in zip(feature_imp['Feature'], feature_imp['Value'])
        ]

        tuples = sorted(tuples, key=lambda x: x[1])[-n_top_feats:]

        labels, values = zip(*tuples)
        _, ax = plt.subplots(1, 1, figsize=(20, 30))
        ylocs = np.arange(len(values))
        # 'top', 'bottom', 'center', 'baseline', 'center_baseline'
        ax.barh(ylocs, values, align="center", height=0.4)
        for x, y in zip(values, ylocs):
            ax.text(x + max(values) / 100, y, x, va="center")

        ax.set_yticks(ylocs)
        ax.set_yticklabels(labels)
        xlim = (0, max(values) * 1.1)
        ax.set_xlim(xlim)
        ylim = (-1, len(values))
        ax.set_ylim(ylim)

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
