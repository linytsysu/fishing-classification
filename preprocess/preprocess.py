import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler

from seglearn.base import TS_Data
from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, SegmentX, SegmentXY

from tsfresh import select_features, extract_features
from parameters import fc_parameters

train_path = '../data/hy_round2_train_20200225'

train_df_list = []
for file_name in os.listdir(train_path):
    if file_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(train_path, file_name))
        train_df_list.append(df)

train_df = pd.concat(train_df_list)

train_df['time'] = pd.to_datetime(train_df['time'], format='%m%d %H:%M:%S')

all_df = pd.concat([train_df])

X = []
y = []
id_list = []
for ship_id, group in all_df.groupby('渔船ID'):
    X.append(group[['lat', 'lon', '速度', '方向', 'time']])
    y.append(group['type'].values[0])
    id_list.append(ship_id)
print(len(id_list))

pype = Pype([('segment', SegmentX(width=72, overlap=0.1))])

pype = pype.fit(X, y)

shape_list = []
df_list = []
for ship_id, group in all_df.groupby('渔船ID'):
    sample = group[['lat', 'lon', '速度', '方向', 'time']].values
    transform_result = pype.transform([sample])[0]

    if transform_result.shape[0] == 0:
        seg_df = pd.DataFrame(sample, columns=['lat', 'lon', '速度', '方向', 'time'])
        seg_df['渔船ID'] = len(df_list)
        seg_df['type'] = group['type'].values[0]
        df_list.append(seg_df)
        shape_list.append(1)
    else:
        for seg in transform_result:
            seg_df = pd.DataFrame(seg, columns=['lat', 'lon', '速度', '方向', 'time'])
            seg_df['渔船ID'] = len(df_list)
            seg_df['type'] = group['type'].values[0]
            df_list.append(seg_df)
        shape_list.append(transform_result.shape[0])

new_all_df = pd.concat(df_list, sort=False)
new_all_df.to_csv('help.csv', index=False)
new_all_df = pd.read_csv('help.csv')
df = new_all_df.drop(columns=['type'])
extracted_df = extract_features(df, column_id='渔船ID', column_sort='time',
                                n_jobs=8, kind_to_fc_parameters=fc_parameters)

y = []
for name, group in new_all_df.groupby('渔船ID'):
    y.append(group.iloc[0]['type'])

train_df = extracted_df.iloc[:np.sum(shape_list[:len(train_df_list)])]

y_train = y[:train_df.shape[0]]
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)

train_df['type'] = le.inverse_transform(y_train)

train_df.to_csv('./train.csv')
