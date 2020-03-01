import os
import numpy as np
import pandas as pd
from tsfresh import select_features, extract_features
from tsfresh.utilities.dataframe_functions import impute
import lightgbm as lgb
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


train_path = '/tcdata/hy_round2_train_20200225'
test_path = '/tcdata/hy_round2_testA_20200225'


def get_feature(arr):
    feature = [np.max(arr), np.quantile(arr, 0.9), np.quantile(arr, 0.1),
               np.quantile(arr, 0.75), np.quantile(arr, 0.25), np.mean(arr), np.std(arr),
               np.median(arr),  np.std(arr) / np.mean(arr)]
    feature.append(np.corrcoef(np.array([arr[:-1], arr[1:]]))[0, 1])
    feature.append(skew(arr))
    feature.append(kurtosis(arr))
    return feature


def get_paper_feature():
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

    features = []
    for _, group in all_df.groupby('渔船ID'):
        group = group.sort_values(by=['time'])
        lat = group['lat'].values
        lon = group['lon'].values
        time_ = group['time'].values
        dire = group['方向'].values

        speed_list = []
        for i in range(lat.shape[0]):
            if i == 0:
                continue
            hour = (time_[i] - time_[i-1]) / np.timedelta64(1,'h')
            dist = geodesic((lat[i-1], lon[i-1]), (lat[i ], lon[i]))
            speed_list.append(dist.km / hour)

        # acc_list = []
        # for i in range(len(speed_list)):
        #     if i == 0:
        #         continue
        #     hour = (time_[i] - time_[i-1]) / np.timedelta64(1,'h')
        #     acc = (speed_list[i] - speed_list[i-1]) / hour
        #     acc_list.append(acc)

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

        features.append(np.concatenate([get_feature(speed_list), angle_feature[:1], turn_feature[:1]]))

    return features[:len(train_df_list)], features[len(train_df_list):]


def tsfresh_extract_features():
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

    df = all_df.drop(columns=['type'])

    extracted_df = extract_features(df, column_id='渔船ID', column_sort='time', n_jobs=8)

    train_df = extracted_df.iloc[:len(train_df_list)]
    test_df = extracted_df.iloc[len(train_df_list):]

    y = []
    for name, group in all_df.groupby('渔船ID'):
        y.append(group.iloc[0]['type'])

    y_train = y[:train_df.shape[0]]
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    impute(train_df)
    filtered_train_df = select_features(train_df, y_train)
    filtered_test_df = test_df[filtered_train_df.columns]

    filtered_train_df['type'] = le.inverse_transform(y_train)

    if not os.path.exists('./feature'):
        os.makedirs('./feature')
    filtered_train_df.to_csv('./feature/train.csv')
    filtered_test_df.to_csv('./feature/test.csv')

    return filtered_train_df, filtered_test_df


def feature_generate_manually():
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

    all_df['x'] = all_df['lat'].values
    all_df['y'] = all_df['lon'].values

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

    xy_cov = []
    unique_x = []
    unique_x_rate = []
    for ship_id, group in all_df.groupby('渔船ID'):
        x = group['x'].values
        y = group['y'].values
        xy_cov.append(group['x'].cov(group['y']))
        unique_x.append(np.unique(x).shape[0])
        unique_x_rate.append(np.unique(y).shape[0] / x.shape[0])

    new_df['xy_cov'] = xy_cov
    new_df['unique_x'] = unique_x
    new_df['unique_x_rate'] = unique_x_rate

    new_df['type'] = all_df.groupby('渔船ID').agg(type=('type', 'first'))['type'].values

    X_train = new_df.drop(columns=['type']).iloc[:len(train_df_list)]
    y_train = new_df.iloc[:len(train_df_list)]['type']

    X_test = new_df.drop(columns=['type']).iloc[len(train_df_list):]

    return X_train, y_train, X_test


def feature_generate_tsfresh():
    train_df = pd.read_csv('./feature/train.csv', index_col=0)
    X_train = train_df.drop(columns=['type'])
    y_train = train_df['type']

    test_df = pd.read_csv('./feature/test.csv', index_col=0)
    X_test = test_df[X_train.columns]

    base_model = lgb.LGBMClassifier(n_estimators=1000, subsample=0.8)
    base_model.fit(X_train.values, y_train)

    selected_columns = X_train.columns[np.argsort(base_model.feature_importances_)[::-1][:50]]
    print(selected_columns)

    X_train = X_train[selected_columns]
    X_test = X_test[selected_columns]

    X_train_manully, _, X_test_manully = feature_generate_manually()

    print(X_train.shape, X_train_manully.shape, X_test.shape, X_test_manully.shape)

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

    for column in list(X_test.columns[X_test.isnull().sum() > 0]):
        mean_val = X_test[column].median()
        X_test[column].fillna(mean_val, inplace=True)

    return X_train, y_train.values, X_test


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


def get_model_v3():
    exported_pipeline = make_pipeline(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        RobustScaler(),
        RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.25, n_estimators=100), step=0.6500000000000001),
        StandardScaler(),
        GradientBoostingClassifier(learning_rate=0.5, max_depth=9, max_features=0.05, min_samples_leaf=18, min_samples_split=3, n_estimators=100, subsample=0.9000000000000001)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)
    return exported_pipeline


if __name__ == "__main__":
    X_paper_train, X_paper_test = get_paper_feature()

    # 生成特征文件
    tsfresh_extract_features()

    X_train_tsfresh, y_train, X_test_tsfresh = feature_generate_tsfresh()
    X_train = np.concatenate([X_train_tsfresh.values, X_paper_train], axis=1)
    X_test = np.concatenate([X_test_tsfresh.values, X_test_train], axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    X_train = imputer.fit_transform(pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan).values)
    X_test = imputer.fit_transform(pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan).values)

    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    kf = KFold(n_splits=5, random_state=2019, shuffle=True)
    model_v1_list = []
    score_v1_list = []
    for train_index, test_index in kf.split(X_train):
        model_v1 = get_model()
        eval_set = (X_train[test_index], y_train[test_index])
        model_v1.fit(X_train[train_index], y_train[train_index])
        model_v1_list.append(model_v1)
        score_v1_list.append(f1_score(y_train[test_index], model_v1.predict(X_train[test_index]), average='macro'))
    print(score_v1_list)
    print(np.mean(score_v1_list), np.std(score_v1_list))

    kf = KFold(n_splits=5, random_state=22, shuffle=True)
    model_v2_list = []
    score_v2_list = []
    for train_index, test_index in kf.split(X_train):
        model_v2 = get_model_v2()
        eval_set = (X_train[test_index], y_train[test_index])
        model_v2.fit(X_train[train_index], y_train[train_index])
        model_v2_list.append(model_v2)
        score_v2_list.append(f1_score(y_train[test_index], model_v2.predict(X_train[test_index]), average='macro'))
    print(score_v2_list)
    print(np.mean(score_v2_list), np.std(score_v2_list))

    kf = KFold(n_splits=5, random_state=22, shuffle=True)
    model_v3_list = []
    score_v3_list = []
    for train_index, test_index in kf.split(X_train):
        model_v3 = get_model_v3()
        eval_set = (X_train[test_index], y_train[test_index])
        model_v3.fit(X_train[train_index], y_train[train_index])
        model_v3_list.append(model_v3)
        score_v3_list.append(f1_score(y_train[test_index], model_v3.predict(X_train[test_index]), average='macro'))
    print(score_v3_list)
    print(np.mean(score_v3_list), np.std(score_v3_list))

    result_list = []
    for model in model_v1_list:
        result = model.predict_proba(X_test)
        result_list.append(result)

    for model in model_v2_list:
        result = model.predict_proba(X_test)
        result_list.append(result)

    for model in model_v3_list:
        result = model.predict_proba(X_test)
        result_list.append(result)

    result = np.argmax(np.sum(np.array(result_list), axis=0) / 15, axis=1)

    result = le.inverse_transform(result)
    pd.DataFrame(result, index=X_test_tsfresh.index).to_csv('./result.csv', header=None)
