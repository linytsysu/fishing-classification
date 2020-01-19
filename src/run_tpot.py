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


def feature_generate_tsfresh():
    train_df = pd.read_csv('./train.csv')
    X_train = train_df.drop(columns=['type'])
    y_train = train_df['type']

    test_df = pd.read_csv('./test.csv')
    X_test = test_df[X_train.columns]

    base_model =  lgb.LGBMClassifier(n_estimators=1000, subsample=0.8)
    base_model.fit(X_train.values, y_train)

    selected_columns = X_train.columns[np.argsort(base_model.feature_importances_)[::-1][:20]]
    print(selected_columns)

    X_train = X_train[selected_columns].values
    X_test = X_test[selected_columns].values

    return X_train, y_train, X_test


if __name__ == "__main__":
    X_train_tsfresh, y_train, X_test_tsfresh = feature_generate_tsfresh()

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=5, scoring='f1_macro',
                                        random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train_tsfresh, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
