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



if __name__ == "__main__":
    train_df = pd.read_csv('./new_train.csv', index_col=0)
    X_train = train_df.drop(columns=['type']).values
    y_train = train_df['type'].values

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    pipeline_optimizer = TPOTClassifier(generations=100, population_size=100, cv=5, scoring='f1_macro',
                                        random_state=42, verbosity=2, n_jobs=8)
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export('tpot_exported_pipeline.py')
