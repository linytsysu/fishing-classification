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
from parameters import fc_parameters_v2

from geopy.distance import geodesic
from scipy.stats import skew, kurtosis

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.impute import SimpleImputer
from tpot.builtins import StackingEstimator, ZeroCount
from xgboost import XGBClassifier


def generate_result():
    train_path = '/tcdata/hy_round2_train_20200225'
    test_path = '/tcdata/hy_round2_testB_20200312'

    train_df_list = []
    for file_name in os.listdir(train_path):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(train_path, file_name))
            train_df_list.append(df)

    test_df_list = []
    for file_name in os.listdir(test_path):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(test_path, file_name))
            test_df_list.append(df)

    train_df = pd.concat(train_df_list)
    test_df = pd.concat(test_df_list)

    train_df['time'] = pd.to_datetime(train_df['time'], format='%m%d %H:%M:%S')
    test_df['time'] = pd.to_datetime(test_df['time'], format='%m%d %H:%M:%S')

    all_df = pd.concat([train_df, test_df])

    X = []
    y = []
    id_list = []
    for ship_id, group in all_df.groupby('渔船ID'):
        X.append(group[['lat', 'lon', '速度', '方向', 'time']])
        y.append(group['type'].values[0])
        id_list.append(ship_id)
    print(len(id_list))

    pype = Pype([('segment', SegmentX(width=72, overlap=0.0))])

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
                                    n_jobs=8, kind_to_fc_parameters=fc_parameters_v2)

    new_df = new_all_df.groupby('渔船ID').agg(x_min=('lat', 'min'), x_max=('lat', 'max'),
                y_min=('lon', 'min'), y_max=('lon', 'max'))
    extracted_df['x_max-x_min'] = new_df['x_max'] - new_df['x_min']
    extracted_df['y_max-y_min'] = new_df['y_max'] - new_df['y_min']
    extracted_df['x_max-y_min'] = new_df['x_max'] - new_df['y_min']
    extracted_df['y_max-x_min'] = new_df['y_max'] - new_df['x_min']
    extracted_df['slope'] = extracted_df['y_max-y_min'] / np.where(extracted_df['x_max-x_min']==0, 0.001, extracted_df['x_max-x_min'])
    extracted_df['area'] = extracted_df['x_max-x_min'] * extracted_df['y_max-y_min']


    def get_feature(arr):
        feature = [np.max(arr), np.quantile(arr, 0.9), np.quantile(arr, 0.1),
                np.quantile(arr, 0.75), np.quantile(arr, 0.25), np.mean(arr), np.std(arr),
                np.median(arr),  np.std(arr) / np.mean(arr)]
        feature.append(np.corrcoef(np.array([arr[:-1], arr[1:]]))[0, 1])
        feature.append(skew(arr))
        feature.append(kurtosis(arr))
        return feature


    features = []
    for _, group in new_all_df.groupby('渔船ID'):
        group = group.sort_values(by=['time'])
        lat = group['lat'].values
        lon = group['lon'].values
        time_ = pd.to_datetime(group['time'], format='%Y-%m-%d %H:%M:%S').values
        dire = group['方向'].values

        speed_list = []
        for i in range(lat.shape[0]):
            if i == 0:
                continue
            hour = (time_[i] - time_[i-1]) / np.timedelta64(1,'h')
            dist = geodesic((lat[i-1], lon[i-1]), (lat[i ], lon[i]))
            speed_list.append(dist.km / hour)

        c = np.sum(np.cos(dire / 180 * np.pi)) / group.shape[0]
        s = np.sum(np.sin(dire / 180 * np.pi)) / group.shape[0]
        r = np.sqrt(c ** 2 + s ** 2)
        theta = np.arctan(s / c)
        angle_feature = [r, theta, np.sqrt(-2 * np.log(r))]

        turn_list = []
        for i in range(dire.shape[0]):
            if i == 0:
                continue
            turn = 1 - np.cos(dire[i-1] / 180 * np.pi - dire[i] / 180 * np.pi)
            turn_list.append(turn * np.pi)
        turn_list = np.array(turn_list)
        c = np.sum(np.cos(turn_list)) / (group.shape[0] - 1)
        s = np.sum(np.sin(turn_list)) / (group.shape[0] - 1)
        r = np.sqrt(c ** 2 + s ** 2)
        theta = np.arctan(s / c)
        turn_feature =  [r, theta, np.sqrt(-2 * np.log(r))]

        features.append(np.concatenate([get_feature(speed_list),
                                        angle_feature[:1], turn_feature[:1]]))

    extracted_df_ = pd.concat([pd.DataFrame(np.array(features)), extracted_df], axis=1)

    y = []
    for _, group in new_all_df.groupby('渔船ID'):
        y.append(group.iloc[0]['type'])

    train_df = extracted_df_.iloc[:np.sum(shape_list[:len(train_df_list)])]
    test_df = extracted_df_.iloc[np.sum(shape_list[:len(train_df_list)]):]

    y_train = y[:train_df.shape[0]]
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    train_df['type'] = le.inverse_transform(y_train)

    train_df.to_csv('./train.csv')
    test_df.to_csv('./test.csv')

    train_df = pd.read_csv('./train.csv', index_col=0)
    X_train = train_df.drop(columns=['type']).values
    y_train = train_df['type'].values

    test_df = pd.read_csv('./test.csv', index_col=0)
    X_test = test_df.values

    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train = imputer.fit_transform(pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan).values)
    X_test = imputer.fit_transform(pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan).values)

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)


    def get_model():
        exported_pipeline = make_pipeline(
            SelectPercentile(score_func=f_classif, percentile=48),
            StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=0.01, fit_intercept=False, l1_ratio=0.25, learning_rate="invscaling", loss="modified_huber", penalty="elasticnet", power_t=10.0)),
            ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.6000000000000001, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
        )

        set_param_recursive(exported_pipeline.steps, 'random_state', 42)
        return exported_pipeline


    def get_model_v2():
        exported_pipeline = make_pipeline(
            make_union(
                make_pipeline(
                    make_union(
                        FunctionTransformer(copy),
                        FunctionTransformer(copy)
                    ),
                    SelectPercentile(score_func=f_classif, percentile=18)
                ),
                FunctionTransformer(copy)
            ),
            StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="hinge", penalty="elasticnet", power_t=0.1)),
            VarianceThreshold(threshold=0.05),
            ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.55, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
        )
        set_param_recursive(exported_pipeline.steps, 'random_state', 42)
        return exported_pipeline

    def get_data(shape_idx):
        start_idx = int(np.sum(shape_list[:shape_idx]))
        end_idx = start_idx + shape_list[shape_idx]
        if shape_idx < len(train_df_list):

            return X_train[start_idx: end_idx], y_train[start_idx: end_idx]
        else:
            return X_test[start_idx: end_idx], None

    kf = KFold(n_splits=5, random_state=2019, shuffle=True)

    model_v1_list = []
    score_v1_list = []
    for train_index, test_index in kf.split(shape_list[:len(train_df_list)]):
        train_data = []
        y_data = []
        for idx in train_index:
            data = get_data(idx)
            train_data.append(data[0])
            y_data.append(data[1])
        train_data = np.concatenate(train_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        model_v1 = get_model()
        model_v1.fit(train_data, y_data)
        model_v1_list.append(model_v1)

        y_true = []
        y_pred = []
        for idx in test_index:
            data = get_data(idx)
            proba = model_v1.predict_proba(data[0])
            pred = np.argmax(np.sum(proba, axis=0) / proba.shape[0])
            y_pred.append(pred)
            y_true.append(data[1][0])
        score = f1_score(y_pred, y_true, average='macro')
        score_v1_list.append(score)

    print(score_v1_list)
    print(np.mean(score_v1_list))

    kf = KFold(n_splits=5, random_state=22, shuffle=True)

    model_v2_list = []
    score_v2_list = []
    for train_index, test_index in kf.split(shape_list[:len(train_df_list)]):
        train_data = []
        y_data = []
        for idx in train_index:
            data = get_data(idx)
            train_data.append(data[0])
            y_data.append(data[1])
        train_data = np.concatenate(train_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        model_v2 = get_model_v2()
        model_v2.fit(train_data, y_data)
        model_v2_list.append(model_v2)

        y_true = []
        y_pred = []
        for idx in test_index:
            data = get_data(idx)
            proba = model_v2.predict_proba(data[0])
            pred = np.argmax(np.sum(proba, axis=0) / proba.shape[0])
            y_pred.append(pred)
            y_true.append(data[1][0])
        score = f1_score(y_pred, y_true, average='macro')
        score_v2_list.append(score)

    print(score_v2_list)
    print(np.mean(score_v2_list))

    kf = KFold(n_splits=5, random_state=22, shuffle=True)

    model_v3_list = []
    score_v3_list = []
    for train_index, test_index in kf.split(shape_list[:len(train_df_list)]):
        train_data = []
        y_data = []
        for idx in train_index:
            data = get_data(idx)
            train_data.append(data[0])
            y_data.append(data[1])
        train_data = np.concatenate(train_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        model_v3 = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=1, min_samples_split=2, n_estimators=100)
        model_v3.fit(train_data, y_data)
        model_v3_list.append(model_v3)

        y_true = []
        y_pred = []
        for idx in test_index:
            data = get_data(idx)
            proba = model_v3.predict_proba(data[0])
            pred = np.argmax(np.sum(proba, axis=0) / proba.shape[0])
            y_pred.append(pred)
            y_true.append(data[1][0])
        score = f1_score(y_pred, y_true, average='macro')
        score_v3_list.append(score)

    print(score_v3_list)
    print(np.mean(score_v3_list))

    kf = KFold(n_splits=5, random_state=22, shuffle=True)

    model_v4_list = []
    score_v4_list = []
    for train_index, test_index in kf.split(shape_list[:len(train_df_list)]):
        train_data = []
        y_data = []
        for idx in train_index:
            data = get_data(idx)
            train_data.append(data[0])
            y_data.append(data[1])
        train_data = np.concatenate(train_data, axis=0)
        y_data = np.concatenate(y_data, axis=0)

        model_v4 = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.6000000000000001, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
        model_v4.fit(train_data, y_data)
        model_v4_list.append(model_v4)

        y_true = []
        y_pred = []
        for idx in test_index:
            data = get_data(idx)
            proba = model_v4.predict_proba(data[0])
            pred = np.argmax(np.sum(proba, axis=0) / proba.shape[0])
            y_pred.append(pred)
            y_true.append(data[1][0])
        score = f1_score(y_pred, y_true, average='macro')
        score_v4_list.append(score)

    print(score_v4_list)
    print(np.mean(score_v4_list))

    pred = []
    for i in range(len(train_df_list), len(shape_list)):
        start_idx = int(np.sum(shape_list[len(train_df_list):i]))
        sample = X_test[start_idx: start_idx+shape_list[i]]
        result = []
        for model in model_v1_list:
            result.append(np.sum(model.predict_proba(sample), axis=0) / shape_list[i])

        for model in model_v2_list:
            result.append(np.sum(model.predict_proba(sample), axis=0) / shape_list[i])

        for model in model_v3_list:
            result.append(np.sum(model.predict_proba(sample), axis=0) / shape_list[i])

        for model in model_v4_list:
            result.append(np.sum(model.predict_proba(sample), axis=0) / shape_list[i])

        pred.append(np.sum(result, axis=0) / 20)

    pd.DataFrame(pred, index=id_list[len(train_df_list):]).to_csv('./probaresult.csv', header=None)
