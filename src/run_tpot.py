#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score
import lightgbm as lgb
from tpot import TPOTClassifier
from sklearn import preprocessing


def feature_generate_manually():
    train_path = '../data/hy_round1_train_20200102'
    test_path = '../data/hy_round1_testA_20200102'

    train_df_list = []
    for file_name in os.listdir(train_path):
        df = pd.read_csv(os.path.join(train_path, file_name))
        train_df_list.append(df)

    test_df_list = []
    for file_name in os.listdir(test_path):
        df = pd.read_csv(os.path.join(test_path, file_name))
        test_df_list.append(df)

    train_df = pd.concat(train_df_list)
    test_df = pd.concat(test_df_list)

    train_df['time'] = pd.to_datetime(train_df['time'], format='%m%d %H:%M:%S')
    test_df['time'] = pd.to_datetime(test_df['time'], format='%m%d %H:%M:%S')

    all_df = pd.concat([train_df, test_df], sort=False)

    new_df = all_df.groupby('渔船ID').agg(x_min=('x', 'min'), x_max=('x', 'max'), x_mean=('x', 'mean'), x_std=('x', 'std'), x_skew=('x', 'skew'), x_sum=('x', 'sum'),
                y_min=('y', 'min'), y_max=('y', 'max'), y_mean=('y', 'mean'), y_std=('y', 'std'), y_skew=('y', 'skew'), y_sum=('y', 'sum'),
                v_min=('速度', 'min'), v_max=('速度', 'max'), v_mean=('速度', 'mean'), v_std=('速度', 'std'), v_skew=('速度', 'skew'), v_sum=('速度', 'sum'),
                d_min=('方向', 'min'), d_max=('方向', 'max'), d_mean=('方向', 'mean'), d_std=('方向', 'std'), d_skew=('方向', 'skew'), d_sum=('方向', 'sum'))
    new_df['x_max-x_min'] = new_df['x_max'] - new_df['x_min']
    new_df['y_max-y_min'] = new_df['y_max'] - new_df['y_min']
    new_df['x_max-y_min'] = new_df['x_max'] - new_df['y_min']
    new_df['y_max-x_min'] = new_df['y_max'] - new_df['x_min']

    new_df['slope'] = new_df['y_max-y_min'] / np.where(new_df['x_max-x_min']==0, 0.001, new_df['x_max-x_min'])
    new_df['area'] = new_df['x_max-x_min'] * new_df['y_max-y_min']

    new_df['type'] = all_df.groupby('渔船ID').agg(type=('type', 'first'))['type'].values

    X_train = new_df.drop(columns=['type']).iloc[:7000]
    y_train = new_df.iloc[:7000]['type']

    X_test = new_df.drop(columns=['type']).iloc[7000:]

    return X_train, y_train, X_test


def feature_generate_tsfresh():
    train_df = pd.read_csv('./train.csv')
    X_train = train_df.drop(columns=['type'])
    y_train = train_df['type']

    test_df = pd.read_csv('./test.csv')
    X_test = test_df[X_train.columns]

    base_model = lgb.LGBMClassifier(n_estimators=1000, subsample=0.8)
    base_model.fit(X_train.values, y_train)

    selected_columns = X_train.columns[np.argsort(base_model.feature_importances_)[::-1][:24]]
    print(selected_columns)

    X_train = X_train[selected_columns]
    X_test = X_test[selected_columns]

    X_train_manully, _, X_test_manully = feature_generate_manually()

    X_train['x_max-x_min'] = X_train_manully['x_max-x_min'].values
    X_test['x_max-x_min'] = X_test_manully['x_max-x_min'].values
    X_train['x_max-y_min'] = X_train_manully['x_max-y_min'].values
    X_test['x_max-y_min'] = X_test_manully['x_max-y_min'].values
    X_train['y_max-x_min'] = X_train_manully['y_max-x_min'].values
    X_test['y_max-x_min'] = X_test_manully['y_max-x_min'].values
    X_train['y_max-y_min'] = X_train_manully['y_max-y_min'].values
    X_test['y_max-y_min'] = X_test_manully['y_max-y_min'].values

    X_train['slope'] = X_train_manully['slope'].values
    X_test['slope'] = X_test_manully['slope'].values
    X_train['area'] = X_train_manully['area'].values
    X_test['area'] = X_test_manully['area'].values

    return X_train.values, y_train.values, X_test.values


if __name__ == "__main__":
    X_train_tsfresh, y_train, X_test_tsfresh = feature_generate_tsfresh()

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=5, scoring='f1_macro',
                                        random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train_tsfresh, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
