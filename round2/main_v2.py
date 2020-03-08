import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from seglearn.base import TS_Data
from seglearn.datasets import load_watch
from seglearn.pipe import Pype
from seglearn.transform import FeatureRep, SegmentX, SegmentXY

train_path = '/tcdata/hy_round2_train_20200225'
test_path = '/tcdata/hy_round2_testA_20200225'

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

X = []
y = []
id_list = []
for ship_id, group in all_df.groupby('渔船ID'):
    X.append(group[['lat', 'lon', '方向', '速度']].values)
    y.append(group['type'].values[0])
    id_list.append(int(ship_id))

le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y[:len(train_df_list)])

X_train = X[:len(train_df_list)]
X_test = X[len(train_df_list):]

kf = KFold(n_splits=5, random_state=42, shuffle=True)
model_v1_list = []
score_v1_list = []
for train_index, test_index in kf.split(X_train):
    model_v1 = Pype([('segment', SegmentX(width=10)),
                        ('features', FeatureRep()),
                        ('scaler', StandardScaler()),
                        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])

    model_v1.fit(np.array(X_train)[train_index], y_train[train_index])

    model_v1_list.append(model_v1)

    y_pred = []
    for test_sample in np.array(X_train)[test_index]:
        result = model_v1.predict_proba([test_sample])
        pred = np.argmax(np.sum(result, axis=0) / result.shape[0])
        y_pred.append(pred)
    score_v1_list.append(f1_score(y_train[test_index], y_pred, average='macro'))

print(score_v1_list)
print(np.mean(score_v1_list), np.std(score_v1_list))

result_list = []
for model in model_v1_list:
    result = []
    for test_sample in X_test:
        sample_result = model_v1.predict_proba([test_sample])
        proba = np.sum(sample_result, axis=0) / sample_result.shape[0]
        result.append(proba)
    result_list.append(result)

result = np.argmax(np.sum(np.array(result_list), axis=0) / 5, axis=1)

result = le.inverse_transform(result)
pd.DataFrame(result, index=id_list[len(train_df_list):]).to_csv('./result.csv', header=None)
