### IMPORTS



import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pickle
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, ParameterGrid, ShuffleSplit
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn import preprocessing
import random
from sklearn.cluster import KMeans
from sklearn import mixture
import math as mt


from sklearn.neighbors import NearestNeighbors

from hyperopt.pyll import scope

from sklearn.model_selection import (cross_val_score, train_test_split,
                                     GridSearchCV, RandomizedSearchCV)


import os
import logging
# Let OpenMP use 4 threads to evaluate models - may run into errors
# if this is not set. Should be set before hyperopt import.
os.environ['OMP_NUM_THREADS'] = '32'

import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def check_test_data(data_test):
    for column in ['WELL', 'DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC', 'GROUP', 'FORMATION', \
       'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', \
       'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', \
       'ROPA', 'RXO']:
        assert column in data_test.columns, "Column {} is expected to be in the test data".format(column)
    print("All expected columns are present in the test data")

    well_location = data_test.groupby('WELL')[['X_LOC', 'Y_LOC']].median()
    if well_location.isna().any().any():
        print('Location information is totally missing for some of the test wells!')
    # assert well_location.isna().any().any() == False

    for column in ['WELL', 'DEPTH_MD', 'GR']:
        assert data_test[column].isna().any() == False, "No missing values in the {} column are expected".format(column)

    print('Test data is OK')

def build_model_and_predict(data_test, output_submission_file = 'test_script.csv', n_steps = 150, hyperopt_niters = 10, is_impute = False, n_iter_imp = 1000):
    print("Shape of test data: {}".format(data_test.shape))
    print(data_test.info())
    check_test_data(data_test)

    def score(y_true, y_pred):
        S = 0.0
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        for i in range(0, y_true.shape[0]):
            S -= A[y_true[i], y_pred[i]]
        return S/y_true.shape[0]


    lithology_keys = {30000: 'Sandstone',
                    65030: 'Sandstone/Shale',
                    65000: 'Shale',
                    80000: 'Marl',
                    74000: 'Dolomite',
                    70000: 'Limestone',
                    70032: 'Chalk',
                    88000: 'Halite',
                    86000: 'Anhydrite',
                    99000: 'Tuff',
                    90000: 'Coal',
                    93000: 'Basement'}

    lithology_numbers = {30000: 0,
                    65030: 1,
                    65000: 2,
                    80000: 3,
                    74000: 4,
                    70000: 5,
                    70032: 6,
                    88000: 7,
                    86000: 8,
                    99000: 9,
                    90000: 10,
                    93000: 11}

    A = np.load('penalty_matrix.npy')

    def feval_f(y_pred, dset):
        y_true = dset.get_label()
        y_pred1=y_pred.reshape(12,-1)
        y_v=np.argmax(y_pred1,axis=0)
        score_v=-score(y_true, y_v)
        return('score', score_v, False)

    ### READ DATA
    data_train = pd.read_csv('train.csv', sep=';')
    print("Shape of train data is: {}".format(data_train.shape))
    print(data_train.info())
    # data_test = pd.read_csv('test.csv', sep=';')
    data_total = pd.concat((data_train, data_test), axis=0)
    print("Shape of total data is: {}".format(data_total.shape))
    print(data_total.info())
    data_total.FORMATION=data_total.FORMATION.astype('category').cat.codes
    data_total.GROUP=data_total.GROUP.astype('category').cat.codes

    ### DATA IMPUTATION

    if is_impute:
        def lgb_r2_score(yhat, dset):
            y_true = dset.get_label()
        #     print(y_true.shape)
            return 'r2', r2_score(y_true, yhat), True

        
        def get_crossval(data_, n_folds=5, random_state=42):
            print('Simple crossval')
            data = data_.copy()
            cv = GroupKFold(n_splits=n_folds)
            cv_labels = data['WELL'].values

            total_test_idx = set()
            for train_index, test_index in cv.split(data['WELL'], y=None, groups=cv_labels):
                total_test_idx = total_test_idx.union(set(test_index))
            print('Total number of records and total number of test records: {} and {}'.format(len(data), len(total_test_idx)))
            
            return cv, cv_labels



        # def get_crossval(data_, n_folds=5, random_state=42):
        #     data = data_.copy()
            
        #     well_location = data.groupby('WELL')[['X_LOC', 'Y_LOC']].median()
        #     print("well location shape: {}".format(well_location.shape))
        #     if well_location.isna().any().any():
        #         print('Location information is totally missing for some wells!')
        #         well_location_median = well_location.median()
        #         well_location_median = {x: y for x,y in zip(well_location_median.index, well_location_median.values)}
        #         well_location.fillna(well_location_median, inplace=True)

        #     # assert well_location.isna().any().any() == False

        #     model = mixture.GaussianMixture(n_folds * 2, covariance_type='full', random_state=3425).fit(well_location)
        #     print('well cluster model is built')
        #     well_location['cluster'] = model.predict(well_location)
        #     print("well location shape after clustering: {}".format(well_location.shape))
        #     print(well_location.info())
        #     if well_location['cluster'].isna().any():
        #         print('some of the wells we not assigned to any cluster')

        #     well_location_map = {x: y for x, y in zip(well_location.index, well_location.cluster)}
            
        #     # define cvlabels
        #     cv_labels = pd.Series(np.zeros(len(data), dtype=int), index=data.index)
        #     data['cluster'] = data['WELL'].map(well_location_map)
        #     if data['cluster'].isna().any():
        #         data['cluster'] = data['cluster'].fillna(0)
        #         print("some of the data records were not clustered!")

        #     well_info = data[['WELL', 'cluster']].drop_duplicates()
        #     if well_info.isna().any().any():
        #         print("missing values in well_info")
        #         well_info['cluster'] = well_info['cluster'].fillna(0)
        #     print(well_info.info())
            
        #     if set(well_info['WELL']) != set(data['WELL']):
        #         print("well_info and data WELL sets are different")
    
        #     cv = StratifiedKFold(n_splits=n_folds, random_state=random_state)
            
        #     for idx, (_, test_wells_idx) in enumerate(cv.split(well_info['WELL'], well_info['cluster'])):
        #         test_wells = well_info.iloc[test_wells_idx]['WELL'].values
        #         cv_labels[data['WELL'].apply(lambda x: x in test_wells)] = int(idx)
            
        #     cv=GroupKFold(n_splits=n_folds)
            
        #     # Now check the crossval
            
            
        #     total_test_idx = set()
        #     for train_index, test_index in cv.split(data['WELL'], y=None, groups=cv_labels):
        #         total_test_idx = total_test_idx.union(set(test_index))
        #     print('Total number of records and total number of test records: {} and {}'.format(len(data), len(total_test_idx)))
            
        #     return cv, cv_labels
        
        cols_to_inpute = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB',  
                        'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS', 
                        'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
        # cols_to_impute = list(set(cols_to_inpute).difference(set(drop_columns)))
        add_cols = ['GR', 'GROUP','FORMATION', 'X_LOC', 'Y_LOC', 'Z_LOC'] #
        models_to_imp = {}

        data_total.reset_index(inplace=True, drop=True)
        print("total data shape after index reset: {}".format(data_total.shape))

        cv, cv_labels = get_crossval(data_total, n_folds=5)
        
        #### CALI

        parameters = {
            'objective': 'regression', 
            'metric': 'mse', 
            'feature_fraction': 0.25,
            'bagging_fraction': 0.01,
            'bagging_freq': 1,
            'n_estimators': n_iter_imp,
            'max_depth': 5,
            'num_leaves': 15,
            'learning_rate': 0.04,
            'verbose': -1,
            'random_seed': 42,
            'min_data_per_group': 1000,
            'bagging_freq': 1
        }


        # cv with index
        col = 'CALI'

        print('Build models for ', col)

        # create df for model
        idx = data_total[col].isna()==False
        df = data_total[idx]
        cv_labels_ = cv_labels[idx]

        # print(df.shape)
        # print(len(cv_labels_))
        kf = cv

        X = df[cols_to_inpute+add_cols].drop([col], axis = 1)
        y = df[col]
        # print(X.shape)
        # print(len(y))
        cat_feat = list(set(['FORMATION','GROUP']).intersection(set(add_cols)))

        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        cvresults = lgb.cv(parameters,
                        train_set=dataset,
                        folds=cv.split(X, y, groups=cv_labels_),
                        categorical_feature=cat_feat,
                        verbose_eval=200,
                        eval_train_metric=True, feval=lgb_r2_score)
        print(cvresults.keys)
        print("{} r2: {} + {}".format(col, np.max(cvresults['valid r2-mean']), cvresults['valid r2-stdv'][np.argmax(cvresults['valid r2-mean'])]))
        parameters['n_estimators'] = np.argmax(cvresults['valid r2-mean'])+1
        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        model = lgb.train(parameters,
                        train_set=dataset,
                        categorical_feature=cat_feat,
                        verbose_eval=200)

        models_to_imp[col] = model

        parameters = {
            'objective': 'regression', 
            'metric': 'mse', 
            'feature_fraction': 0.1,
            'bagging_fraction': 0.03,
            'n_estimators': n_iter_imp,
            'max_depth': 4,
            'num_leaves': 7,
            'learning_rate': 0.04,
            'verbose': -1,
            'random_seed': 42,
            'min_data_per_group': 1000,
            'bagging_freq': 1
        }


        # cv with index
        col = 'RHOB'

        print('Build models for ', col)

        # create df for model
        idx = data_total[col].isna()==False
        df = data_total[idx]
        cv_labels_ = cv_labels[idx]

        # print(df.shape)
        # print(len(cv_labels_))
        kf = cv

        X = df[cols_to_inpute+add_cols].drop([col], axis = 1)
        y = df[col]
        # print(X.shape)
        # print(len(y))
        dataset = lgb.Dataset(X, y, categorical_feature=['FORMATION','GROUP'])

        cvresults = lgb.cv(parameters,
                        train_set=dataset,
                        folds=cv.split(X, y, groups=cv_labels_),
                        categorical_feature=['GROUP', 'FORMATION'],
                        verbose_eval=200,
                        eval_train_metric=True, feval=lgb_r2_score)
        print("{} r2: {} + {}".format(col, np.max(cvresults['valid r2-mean']), cvresults['valid r2-stdv'][np.argmax(cvresults['valid r2-mean'])]))
        parameters['n_estimators'] = np.argmax(cvresults['valid r2-mean'])+1
        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        model = lgb.train(parameters,
                        train_set=dataset,
                        categorical_feature=cat_feat,
                        verbose_eval=200)

        models_to_imp[col] = model

        parameters = {
            'objective': 'regression', 
            'metric': 'mse', 
            'feature_fraction': 0.1,
            'bagging_fraction': 0.03,
            'n_estimators': n_iter_imp,
            'max_depth': 4,
            'num_leaves': 7,
            'learning_rate': 0.04,
            'verbose': -1,
            'random_seed': 42,
            'min_data_per_group': 1000,
            'bagging_freq': 1
        }


        # cv with index
        col = 'NPHI'

        print('Build models for ', col)

        # create df for model
        idx = data_total[col].isna()==False
        df = data_total[idx]
        cv_labels_ = cv_labels[idx]

        # print(df.shape)
        # print(len(cv_labels_))
        kf = cv

        X = df[cols_to_inpute+add_cols].drop([col], axis = 1)
        y = df[col]
        # print(X.shape)
        # print(len(y))
        dataset = lgb.Dataset(X, y, categorical_feature=['FORMATION','GROUP'])

        cvresults = lgb.cv(parameters,
                        train_set=dataset,
                        folds=cv.split(X, y, groups=cv_labels_),
                        categorical_feature=['GROUP', 'FORMATION'],
                        verbose_eval=200,
                        eval_train_metric=True, feval=lgb_r2_score)
        print("{} r2: {} + {}".format(col, np.max(cvresults['valid r2-mean']), cvresults['valid r2-stdv'][np.argmax(cvresults['valid r2-mean'])]))
        parameters['n_estimators'] = np.argmax(cvresults['valid r2-mean'])+1
        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        model = lgb.train(parameters,
                        train_set=dataset,
                        categorical_feature=cat_feat,
                        verbose_eval=200)

        models_to_imp[col] = model

        parameters = {
            'objective': 'regression', 
            'metric': 'mse', 
            'feature_fraction': 0.1,
            'bagging_fraction': 0.03,
            'n_estimators': n_iter_imp,
            'max_depth': 4,
            'num_leaves': 7,
            'learning_rate': 0.04,
            'verbose': -1,
            'random_seed': 42,
            'min_data_per_group': 1000,
            'bagging_freq': 1
        }


        # cv with index
        col = 'DTC'

        print('Build models for ', col)

        # create df for model
        idx = data_total[col].isna()==False
        df = data_total[idx]
        cv_labels_ = cv_labels[idx]

        # print(df.shape)
        # print(len(cv_labels_))
        kf = cv

        X = df[cols_to_inpute+add_cols].drop([col], axis = 1)
        y = df[col]
        # print(X.shape)
        # print(len(y))
        dataset = lgb.Dataset(X, y, categorical_feature=['FORMATION','GROUP'])

        cvresults = lgb.cv(parameters,
                        train_set=dataset,
                        folds=cv.split(X, y, groups=cv_labels_),
                        categorical_feature=['GROUP', 'FORMATION'],
                        verbose_eval=200,
                        eval_train_metric=True, feval=lgb_r2_score)
        print("{} r2: {} + {}".format(col, np.max(cvresults['valid r2-mean']), cvresults['valid r2-stdv'][np.argmax(cvresults['valid r2-mean'])]))
        parameters['n_estimators'] = np.argmax(cvresults['valid r2-mean'])+1
        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        model = lgb.train(parameters,
                        train_set=dataset,
                        categorical_feature=cat_feat,
                        verbose_eval=200)

        models_to_imp[col] = model

        parameters = {
            'objective': 'regression', 
            'metric': 'mse', 
            'feature_fraction': 0.25,
            'bagging_fraction': 0.01,
            'n_estimators': n_iter_imp,
            'max_depth': 5,
            'num_leaves': 15,
            'learning_rate': 0.04,
            'verbose': -1,
            'random_seed': 42,
            'min_data_per_group': 1000,
            'bagging_freq': 1
        }


        # cv with index
        col = 'BS'

        print('Build models for ', col)

        # create df for model
        idx = data_total[col].isna()==False
        df = data_total[idx]
        cv_labels_ = cv_labels[idx]

        # print(df.shape)
        # print(len(cv_labels_))
        kf = cv

        X = df[cols_to_inpute+add_cols].drop([col], axis = 1)
        y = df[col]
        # print(X.shape)
        # print(len(y))
        dataset = lgb.Dataset(X, y, categorical_feature=['FORMATION','GROUP'])

        cvresults = lgb.cv(parameters,
                        train_set=dataset,
                        folds=cv.split(X, y, groups=cv_labels_),
                        categorical_feature=['GROUP', 'FORMATION'],
                        verbose_eval=200,
                        eval_train_metric=True, feval=lgb_r2_score)
        print("{} r2: {} + {}".format(col, np.max(cvresults['valid r2-mean']), cvresults['valid r2-stdv'][np.argmax(cvresults['valid r2-mean'])]))
        parameters['n_estimators'] = np.argmax(cvresults['valid r2-mean'])+1
        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        model = lgb.train(parameters,
                        train_set=dataset,
                        categorical_feature=cat_feat,
                        verbose_eval=200)

        models_to_imp[col] = model

        parameters = {
            'objective': 'regression', 
            'metric': 'mse', 
            'feature_fraction': 0.25,
            'bagging_fraction': 0.03,
            'n_estimators': n_iter_imp,
            'max_depth': 4,
            'num_leaves': 7,
            'learning_rate': 0.04,
            'verbose': -1,
            'random_seed': 42,
            'min_data_per_group': 1000,
            'bagging_freq': 1
        }


        # cv with index
        col = 'DTS'

        print('Build models for ', col)

        # create df for model
        idx = data_total[col].isna()==False
        df = data_total[idx]
        cv_labels_ = cv_labels[idx]

        # print(df.shape)
        # print(len(cv_labels_))
        kf = cv

        X = df[cols_to_inpute+add_cols].drop([col], axis = 1)
        y = df[col]
        # print(X.shape)
        # print(len(y))
        dataset = lgb.Dataset(X, y, categorical_feature=['FORMATION','GROUP'])

        cvresults = lgb.cv(parameters,
                        train_set=dataset,
                        folds=cv.split(X, y, groups=cv_labels_),
                        categorical_feature=['GROUP', 'FORMATION'],
                        verbose_eval=200,
                        eval_train_metric=True, feval=lgb_r2_score)
        print("{} r2: {} + {}".format(col, np.max(cvresults['valid r2-mean']), cvresults['valid r2-stdv'][np.argmax(cvresults['valid r2-mean'])]))
        parameters['n_estimators'] = np.argmax(cvresults['valid r2-mean'])+1
        dataset = lgb.Dataset(X, y, categorical_feature=cat_feat)

        model = lgb.train(parameters,
                        train_set=dataset,
                        categorical_feature=cat_feat,
                        verbose_eval=200)

        models_to_imp[col] = model

        with open('imputation_models_2.pickle', 'wb') as handle:
            pickle.dump(models_to_imp, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('imputation_models_2.pickle', 'rb') as handle:
            models_to_imp = pickle.load(handle)

        cols_to_use = ['CALI','RHOB','NPHI','DTC','BS','DTS']
        a = data_total[['WELL']+cols_to_use].groupby(['WELL']).count().reset_index()
        b = data_total[['WELL']+cols_to_use].fillna(-1).groupby(['WELL']).count().reset_index()
        a[cols_to_use] = a[cols_to_use]/b[cols_to_use]

        data_total_ = data_total.copy()

        for col in cols_to_use:
            wells_to_impute = a[a[col] >= -1]['WELL']
            
            X_n = data_total_.loc[(data_total[col].isna())&
                                    (data_total['WELL'].isin(wells_to_impute)), 
                                    cols_to_inpute+add_cols].drop([col], axis=1)
            
            model = models_to_imp[col]
            prediction = model.predict(X_n)

            data_total_.loc[(data_total_[col].isna())&
                                    (data_total_['WELL'].isin(wells_to_impute)),  col] = prediction

        data_total = data_total_

        n_train = len(data_train)
        data_train = data_total.iloc[:n_train]
        data_test = data_total.iloc[n_train:]

        data_train.to_csv('train-imputed-2.csv', index=False)
        data_test.to_csv('test-imputed-2.csv', index=False)

        data_train = pd.read_csv('train-imputed-2.csv')
        data_test = pd.read_csv('test-imputed-2.csv')

        data_total = pd.concat((data_train, data_test), axis=0)

        data_total['FORMATION']=data_total['FORMATION'].astype('category')
        data_total['GROUP']=data_total['GROUP'].astype('category')
        

    ### FEATURE RANGES AND FEATURE GENERATION
    data_total.loc[data_total['SP'] < -998, 'SP'] = 0
    data_total.loc[data_total['ROPA'] < -998, 'ROPA'] = 0
    data_total.loc[data_total['RXO'] < -998, 'RXO'] = 0

    range_maps = {"CALI": [2, 28], 
                "GR": [0, 1000], 
                "RSHA": [0, 2000],
                "RMED": [0, 2000],
                "RDEP": [0, 2000],
                "RXO": [0, 2000],
                "RMIC": [0, 2000],
                "RHOB": [1, 3.2],
                "NPHI": [-0.4, 1.2],
                "DTS": [70, 500],
                "DTC": [50, 300],
                "PEF": [0, 80], 
                "SP": [-300, 530],
                "MUDWEIGHT": [0, 2]}

    for column in data_total:
        if column in range_maps.keys():
            data_total[column] = data_total[column].clip(lower = range_maps[column][0],
                                                                upper = range_maps[column][1])
                                    
    # generate new features
    K1 = 2.65
    K2 = 1.10
    data_total['PHID'] = (data_total['RHOB']-K1)/(K2-K1)
    data_total['DeltaPHI'] = data_total['PHID'] - data_total['NPHI']
    data_total['PHIND'] = (data_total['PHID']+data_total['NPHI'])/2
    # data_total['PHIND_2'] = np.sqrt((data_total['PHID'] ** 2+data_total['NPHI'] ** 2)/2)
    data_total['DCALI'] = data_total['CALI']-data_total['BS']
    data_total['RHOB_NPHI'] = data_total['RHOB'].apply(lambda x: (x - 1.5) / (3 - 1.5)
                                ).clip(0,1) - data_total['NPHI'].apply(lambda x: (x - 0.6) / ( - 0.6)).clip(0,1)

    data_total['DCALI_per'] = data_total['DCALI'].div(data_total['BS'])
    # data_total['Rw'] = data_total['RDEP'].div(data_total['RDEP'].quantile(0.03)).apply(np.sqrt)
    # data_total['Rwt'] = data_total['Rw'].div(data_total['PHIND'])

    # data_total[['RSHA_std', 'RMED_std', 'RDEP_std']] = data_total[['RSHA', 'RMED', 'RDEP']
    #                                                              ].rolling(5, min_periods=1, center=True).std()

    data_total['Vp'] = 10**6/data_total['DTC']
    data_total['Vs'] = 10**6/data_total['DTS']
    data_total['V_ratio'] = data_total['Vp'] / data_total['Vs']
    data_total['lambd'] = data_total['RHOB'] * (data_total['Vp'] ** 2 - 2 * data_total['Vs'] ** 2)
    data_total['mu'] = data_total['RHOB'] * data_total['Vs'] ** 2
    data_total['E'] = (data_total['mu'] * (3 * data_total['Vp'] ** 2 - 4 * data_total['Vs'] ** 2)) / (
                            data_total['Vp'] ** 2 - data_total['Vs'] ** 2)
    data_total['K'] = data_total['RHOB'] * (data_total['Vp'] ** 2 - (4 / 3) * data_total['Vs'] ** 2)
    data_total['v'] = (data_total['Vp'] ** 2 - 2 * data_total['Vs'] ** 2) / (
                            2 * (data_total['Vp'] ** 2 - data_total['Vs'] ** 2))
    data_total['M'] = data_total['RHOB'] * data_total['Vp'] ** 2

    data_total['IMP_p'] = data_total['Vp'] * data_total['RHOB']
    data_total['IMP_s'] = data_total['Vs'] * data_total['RHOB']

    data_total['RC_p'] = (data_total['IMP_p'].shift(-1) - data_total['IMP_p']
                        ) / (data_total['IMP_p'].shift(-1) + data_total['IMP_p'])
    data_total['RC_s'] = (data_total['IMP_s'].shift(-1) - data_total['IMP_s']
                        ) / (data_total['IMP_s'].shift(-1) + data_total['IMP_s'])
    data_total['depth_frac'] = data_total['DEPTH_MD'] / data_total['Z_LOC']

    data_total.head()


    # data_total['IGR'] = np.log1p(data_total['GR'])
    data_total['IGR'] = data_total['GR']
    data_total['ISP'] = data_total['SP']
    for well in data_total.WELL.unique():
    #     print(well)
        idx = data_total.WELL == well
        scaler = RobustScaler()
        scaler_sp = RobustScaler()


        data_total.loc[idx, 'GR'] = scaler.fit_transform(data_total.loc[idx, 'GR'].values.reshape(-1, 1))
        data_total.loc[idx, 'SP'] = scaler_sp.fit_transform(data_total.loc[idx, 'SP'].values.reshape(-1, 1))


    predictors_new = ['GROUP','FORMATION']+\
                    ['X_LOC', 'Y_LOC', 'Z_LOC']+\
                    ['CALI','RSHA','RMED','RDEP','RHOB','NPHI','PEF','DTC','SP','BS',
                    'ROP','DTS','MUDWEIGHT','ROPA','RXO','DCAL']+\
                    ['GR']+\
                    ['PHID','DeltaPHI','PHIND','DCALI']+\
                    ['IGR', 'ISP']+\
                    ['RHOB_NPHI', 'DCALI_per', 'V_ratio']+\
                    ['IMP_p', 'RC_p', 'RC_s', 'depth_frac']
    #                 ['V_ratio', 'lambd', 'mu', 'E', 'K', 'v', 'M'] +\

    ### CHOOSE HOLDOUT WELLS
    well_loc_train = data_train.groupby('WELL')[['X_LOC', 'Y_LOC']].median()
    well_loc_test = data_test.groupby('WELL')[['X_LOC', 'Y_LOC']].median()

    well_loc_train.info()
    well_loc_test.info()
    well_loc_train.dropna(inplace=True)
    well_loc_test.dropna(inplace=True)
    well_loc_train.info()
    well_loc_test.info()

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(well_loc_train)

    distances, idx = nn.kneighbors(well_loc_test)
    idx_pd = pd.DataFrame(idx, columns=['n1', 'n2'])


    indices = list(set(idx_pd['n1'].values))
    for idx, value in idx_pd['n1'].value_counts().iteritems():
        if value > 1:
            indices += list(idx_pd.loc[idx_pd['n1'] == idx, 'n2'].values)

    indices = list(set(indices))

    holdout_wells = [well_loc_train.index[i] for i in set(indices)]
    holdout_wells = np.asarray(holdout_wells)
    if len(holdout_wells) > 15:
        holdout_wells = holdout_wells[:15]


    ### APPLY HYPEROPT TO FIND OK MODEL
    n_train = len(data_train)

    data = data_total.iloc[:n_train]
    data_test = data_total.iloc[n_train:]

    wells = data.WELL.unique()
    working_wells = np.asarray(list(set(wells).difference(set(holdout_wells))))

    holdout_data = data[data.WELL.apply(lambda x: x in holdout_wells)]
    working_data = data[data.WELL.apply(lambda x: x in working_wells)]

    X = working_data[predictors_new]
    X_test = holdout_data[predictors_new]
    y = working_data['FORCE_2020_LITHOFACIES_LITHOLOGY']
    y = y.map(lithology_numbers)
    X, y = shuffle(X, y, random_state=42)
    y_test = holdout_data['FORCE_2020_LITHOFACIES_LITHOLOGY']
    y_test = y_test.map(lithology_numbers)

    X[['FORMATION','GROUP']] = X[['FORMATION','GROUP']].astype('category')
    X_test[['FORMATION','GROUP']] = X_test[['FORMATION','GROUP']].astype('category')




    # -----------------------------------------------------
    #                       SETUP
    # -----------------------------------------------------

    SEED = 42  # Fix the random state to the ultimate answer in life.
    # Initialize logger
    #logging.basicConfig(filename="xgb_hyperopt.log", level=print)

    best_score = 100
    best_params = {}
    with open('best_score.pickle', 'wb') as handle:
        pickle.dump((best_score, best_params), handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    # -----------------------------------------------------
    #                       HYPEROPT
    # -----------------------------------------------------

    def score_model(params):
        print("Training with params: ")
        print(params)
        with open('best_score.pickle', 'rb') as handle:
            best_score, best_params = pickle.load(handle)
        


        train_d = lgb.Dataset(X, label=y, categorical_feature=['FORMATION','GROUP'])
        test_d = lgb.Dataset(X_test, label=y_test, categorical_feature=['FORMATION','GROUP'])
        # to record eval results for plotting
        evals_result = {} 

        model_final = lgb.train(params, train_d,
                                valid_sets=[test_d],
                                evals_result=evals_result,
                                early_stopping_rounds=50,
                                categorical_feature=['FORMATION','GROUP'],
                                verbose_eval=100,
                                feval=feval_f)
        final_score = model_final.best_score['valid_0']['score']

        with open("hyperopt.txt", "a") as myfile:
            myfile.write(str(params) + '\n')
            myfile.write(str(final_score) + '\n')
            myfile.write('\n------------------')
        
        if final_score < best_score:
            best_score = final_score
            best_params = params
            with open('best_score.pickle', 'wb') as handle:
                pickle.dump((best_score, best_params), handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        
        return {'loss': final_score, 'status': STATUS_OK}


    def optimize(
        # trials,
            random_state=SEED):
        """
        This is the optimization function that given a space (space here) of
        hyperparameters and a scoring function (score here),
        finds the best hyperparameters.
        """

        space = {
            'max_depth': scope.int(hp.uniform('max_depth', 5, 15)),
            'subsample': hp.uniform('subsample', 0.03, 1),
            'learning_rate' : hp.loguniform('learning_rate', np.log(0.005), np.log(0.5)) - 0.0001,
            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(0.005), np.log(5)) - 0.0001,
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1), np.log(5)),
            'bagging_freq': hp.choice('bagging_freq', [0, 1]),
            'num_leaves': scope.int(hp.uniform('num_leaves', 10, 128)),
            'n_estimators': 1000,
            'boosting': 'gbdt',
            'objective': 'multiclass',
            'num_class':  12,
            'metric': 'None',
            'is_unbalance': 'true',
    #         'min_data_per_group': 1000,
            'verbose': -1,
            'random_seed': 42,
            
        }

        # Use the fmin function from Hyperopt to find the best hyperparameters
        best = fmin(score_model, space, algo=tpe.suggest,
                    # trials=trials,
                    max_evals=hyperopt_niters)
        return best


    best_hyperparams = optimize(
        # trials
    )

    with open('best_score.pickle', 'rb') as handle:
        best_score, best_params = pickle.load(handle)
        
    print("The best hyperparameters are: ", "\n")
    print(best_params)


    train_d = lgb.Dataset(X, label=y, categorical_feature=['FORMATION','GROUP'])
    test_d = lgb.Dataset(X_test, label=y_test, categorical_feature=['FORMATION','GROUP'])
    # to record eval results for plotting
    evals_result = {} 

    model_final = lgb.train(best_params, train_d,
                            valid_sets=[test_d],
                            evals_result=evals_result,
                            early_stopping_rounds=50,
                            categorical_feature=['FORMATION','GROUP'],
                            verbose_eval=100,
                            feval=feval_f)


    ### Training/Filtering pipeline
    test_pred = model_final.predict(X_test)
    test_pred = np.array([np.argmax(line) for line in test_pred])
    test_score = score(y_test.values, test_pred)

    print('Test score = ', test_score)
    data_score_test = pd.DataFrame({'WELL': holdout_data['WELL'], 'pred': test_pred, 'y': y_test.values})

    d_score = pd.DataFrame(data_score_test.groupby('WELL').apply(lambda x: score(x['y'].values, x['pred'].values)), columns=['score'])
    d_score.reset_index(inplace=True)
    d_score['size'] = data_score_test.groupby('WELL')['y'].agg('count').values


    param_grid = {
        'objective': ['multiclass'],
        'num_class':[12],
        'metric': ['None'],
        'is_unbalance': ['true'],
        'boosting': ['gbdt'],
        'n_estimators': [500],
        'bagging_freq': [1],
        'min_data_per_group': [1000],
        'verbose': [-1],
        
        'colsample_bytree': [0.3,  0.4, 0.45, 0.5, 0.7],
        'subsample': [0.03, 0.1, 0.25, 0.5, 0.7, 0.9, 1],
        
        'max_depth': [5, 7, 10, 15, 20],
        'num_leaves': [16, 1000],
        
        'reg_alpha': [0, 0.2, 0.4, 0.6],
        'reg_lambda': [1, 2, 3],

        'learning_rate': [0.01, 0.02, 0.05],
    #     'boosting': ['gbdt', 'dart'],
        'boosting': ['gbdt'],
        'random_seed': list(range(200)) 
    }



    param_grid_list = list(ParameterGrid(param_grid))

    models = []

    rs = ShuffleSplit(n_splits=n_steps, test_size=.1, random_state=42)
    rs_test = ShuffleSplit(n_splits=n_steps, test_size=.3, random_state=42)
    index = 0
    for (train_index, valid_index), (tr_index, val_index) in zip(rs.split(working_wells), rs_test.split(holdout_wells)):
        index += 1
        print('----------------------------------')
        print("working on model {}".format(index))
        train_wells, valid_wells = working_wells[train_index], working_wells[valid_index]
        tr_wells, test_wells = holdout_wells[tr_index], holdout_wells[val_index]
        
        train_wells = np.asarray(list(train_wells) + list(tr_wells))
        
        d_score_test = d_score[d_score.WELL.apply(lambda x: x in test_wells)]
        ideal_test_score = (d_score_test['score'] * d_score_test['size']).sum() / d_score_test['size'].sum()
        num_test_records = d_score_test['size'].sum()
        
        print("Number of test records: {} \nIdeal test score: {}".format(num_test_records, ideal_test_score))
        if num_test_records < (d_score['size'].sum() * 0.15):
            continue
        
        
        train_data = data[data.WELL.apply(lambda x: x in train_wells)]
        valid_data = data[data.WELL.apply(lambda x: x in valid_wells)]
        test_data = data[data.WELL.apply(lambda x: x in test_wells)]
        
        
        
        X_train = train_data[predictors_new]
        X_train[['FORMATION','GROUP']] = X_train[['FORMATION','GROUP']].astype('category')
        
        X_valid = valid_data[predictors_new]
        X_valid[['FORMATION','GROUP']] = X_valid[['FORMATION','GROUP']].astype('category')
        
        X_test = test_data[predictors_new]
        X_test[['FORMATION','GROUP']] = X_test[['FORMATION','GROUP']].astype('category')
        
        y_train = train_data['FORCE_2020_LITHOFACIES_LITHOLOGY']
        y_train = y_train.map(lithology_numbers)
        
        y_valid = valid_data['FORCE_2020_LITHOFACIES_LITHOLOGY']
        y_valid = y_valid.map(lithology_numbers)
        
        y_test = test_data['FORCE_2020_LITHOFACIES_LITHOLOGY']
        y_test = y_test.map(lithology_numbers)
        
        train_d = lgb.Dataset(X_train, label=y_train, categorical_feature=['FORMATION','GROUP'])
        valid_d = lgb.Dataset(X_valid, label=y_valid, categorical_feature=['FORMATION','GROUP'])
        
        parameters = np.random.choice(param_grid_list)
        model = lgb.train(parameters, train_d,
                                valid_sets=[valid_d],
                                early_stopping_rounds=50,
                                num_boost_round=500,
                                categorical_feature=['FORMATION','GROUP'],
                                verbose_eval=501,
                                feval=feval_f)
        
        
        test_pred = model.predict(X_test)
        test_pred = np.array([np.argmax(line) for line in test_pred])
        test_score = score(y_test.values, test_pred)
        
        print('Test score = ', test_score)
        if test_score > ideal_test_score:
            models.append(model)
            
            with open('models.pickle', 'wb') as handle:
                pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            print('Currently there are {} good models'.format(len(models)))


    ### Predicting on test test
    X_subm_ = data_test[predictors_new]
    # X_subm_.fillna(0, inplace=True)
    X_subm_[['FORMATION','GROUP']] = X_subm_[['FORMATION','GROUP']].astype('category')
    X_subm_.head()


    n_models = len(models)
    n_records = len(X_subm_)
    n_classes = 12

    subm_pred = np.zeros((n_models, n_records))

    for idx, model in enumerate(models):
        subm_pred_model = model.predict(X_subm_)
        subm_pred_model = np.array([np.argmax(line) for line in subm_pred_model])
        subm_pred[idx, :] = subm_pred_model

    subm_pred, _ = np.asarray(scipy.stats.mode(subm_pred, axis=0))
    subm_pred = np.squeeze(subm_pred)
    subm_pred.shape

    category_to_lithology = {y:x for x,y in lithology_numbers.items()}

    prediction_for_submission = np.vectorize(category_to_lithology.get)(subm_pred)
    np.savetxt(output_submission_file, prediction_for_submission, header='lithology', comments='', fmt='%i')
    

if __name__ == '__main__':
    print('Running...')
    # Set the filename of the data we would like to get the predictions for
    TEST_DATA_FILENAME = 'test.csv'
    data_test =  pd.read_csv(TEST_DATA_FILENAME, sep=';')
    build_model_and_predict(data_test, 'softserve_submission.csv', n_steps = 230, hyperopt_niters = 30, is_impute = True, n_iter_imp = 1000)
    # build_model_and_predict(data_test, 'softserve_test_submission.csv', n_steps = 5, hyperopt_niters = 2, is_impute = True, n_iter_imp = 10)
    