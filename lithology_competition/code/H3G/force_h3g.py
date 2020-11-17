import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier, plot_importance
from sklearn.utils import class_weight
from sklearn import mixture
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from glob import glob


def get_score(A, y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S / y_true.shape[0]


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.xlim(-0.5, len(np.unique(y_true)) - 0.5)
    plt.ylim(len(np.unique(y_true)) - 0.5, -0.5)
    np.set_printoptions(precision=2)
    plt.savefig(title + '.png')
    return ax


def plot_feature_importance(clf, title='Feature Importance'):
    fea_imp = pd.DataFrame({'imp': clf.feature_importances_, 'col': clf.feature_names_})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    fea_imp.plot(kind='barh', x='col', y='imp', figsize=(10, 7), legend=None)
    plt.title(title)
    plt.ylabel('Features')
    plt.xlabel('Importance')
    plt.savefig(title + '.png', bbox_inches='tight')


class Model(object):
    def _preprocess(self, features, train=True, add_seq_features=False, add_poly_features=False,
                    add_log_transformation=False):
        wells = features['WELL'].unique()

        interpolate_features = ['X_LOC', 'Y_LOC', 'Z_LOC', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF',
                                'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']

        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(features[['DEPTH_MD', 'Z_LOC']])
        features['Z_LOC'] = imp.transform(features[['DEPTH_MD', 'Z_LOC']])[:, 1]

        for well in wells:
            features.loc[features.WELL == well, interpolate_features] = \
                features.loc[features.WELL == well, interpolate_features].interpolate().bfill('rows').ffill('rows')

        start_time = time.time()
        for w in features.loc[(features.GROUP.isna()), 'WELL'].unique():
            if (max(features.loc[(features.WELL == w) & (features.GROUP.isna())].index) + 1 - min(
                    features.loc[(features.WELL == w) & (features.GROUP.isna())].index)) == len(
                features.loc[(features.WELL == w) & (features.GROUP.isna())]):
                if features.loc[max(features.loc[(features.WELL == w) & (
                features.GROUP.isna())].index) + 1, 'GROUP'] == 'NORDLAND GP.':
                    features.loc[min(features.loc[(features.WELL == w) & (features.GROUP.isna())].index):max(
                        features.loc[(features.WELL == w) & (features.GROUP.isna())].index), 'GROUP'] = 'NORDLAND GP.'
        features.loc[((features.FORMATION.isna()) & (features.GROUP == 'NORDLAND GP.')), 'FORMATION'] = 'Utsira Fm.'

        features['BS'] = round(features.BS, 2)
        BS_list = [12.25, 8.50, 17.50, 26.00, 9.63, 6.00]
        for i in range(len(features)):
            bs_value = features.BS[i]
            if ((~np.isnan(bs_value)) and (bs_value not in BS_list)):
                features.loc[i, 'BS'] = min(BS_list, key=lambda x: abs(x - bs_value))
        BS_dict = {'NORDLAND GP.': 17.50, 'HORDALAND GP.': 12.25, 'ROGALAND GP.': 12.25, 'SHETLAND GP.': 12.25,
                   'CROMER KNOLL GP.': 8.50, 'VIKING GP.': 8.50, 'VESTLAND GP.': 8.50, 'ZECHSTEIN GP.': 12.25,
                   'HEGRE GP.': 8.50, 'ROTLIEGENDES GP.': 12.25, 'TYNE GP.': 8.50, 'BOKNFJORD GP.': 12.25,
                   'DUNLIN GP.': 8.50, 'BAAT GP.': 8.50}
        features.loc[features.BS.isna(), 'BS'] = features[features.BS.isna()].GROUP.map(BS_dict)
        print("--- %s seconds ---" % (time.time() - start_time))

        features.loc[features.DCAL.isna(), 'DCAL'] = features.loc[features.DCAL.isna(), 'CALI'] - features.loc[
            features.DCAL.isna(), 'BS']
        features.loc[features.DTC.isna(), 'DTC'] = 1e6 / ((features.loc[features.DTC.isna(), 'RHOB'] / 0.23) ** 4)

        features_fe = features

        if add_log_transformation:
            log_features = ['RSHA', 'RDEP', 'RMED', 'RXO']
            for feature in log_features:
                features_fe[feature] = np.log(features_fe[feature])

        if add_seq_features:
            start_time = time.time()
            seq_column_names = ['RHOB', 'GR']
            grad_column_names = ['RHOB', 'GR', 'NPHI', 'DTC', 'DRHO']
            seq_length = 3
            features_fe = pd.DataFrame()
            for w in np.unique(wells):
                print(w)
                tmp = features[features['WELL'] == w]
                for i in range(1, seq_length + 1):
                    sf_df = tmp[seq_column_names].shift(i)
                    sf_df.columns = [x + '_sf_' + str(i) for x in seq_column_names]
                    tmp = tmp.join(sf_df).bfill('rows')

                for grad in grad_column_names:
                    tmp[grad + '_grad'] = np.gradient(tmp[grad].rolling(center=False, window=1).mean())
                features_fe = features_fe.append(tmp)
            print("--- %s seconds ---" % (time.time() - start_time))

        features_fe = features_fe.astype({"GROUP": str, "FORMATION": str, "BS": str})
        features_fe.loc[:, ['GROUP', 'FORMATION', 'BS']] = features_fe[['GROUP', 'FORMATION', 'BS']].fillna('unknown')
        features_fe.fillna(features_fe.median(), inplace=True)

        add_rolling_average = True
        if add_rolling_average:
            start_time = time.time()
            list_to_roll = ['GR', 'DTC']
            rolled_list = []
            window_size = [5, 15, 30]
            for w in window_size:
                for val in list_to_roll:
                    features_fe[val + '_rollingmean_' + str(w)] = features_fe.groupby("WELL")[val].apply(
                        lambda x: x.rolling(window=w, center=True).mean())
                    rolled_list.append(val + '_rollingmean_' + str(w))
                    features_fe[val + '_rollingmax_' + str(w)] = features_fe.groupby("WELL")[val].apply(
                        lambda x: x.rolling(window=w, center=True).max())
                    rolled_list.append(val + '_rollingmax_' + str(w))
                    features_fe[val + '_rollingmin_' + str(w)] = features_fe.groupby("WELL")[val].apply(
                        lambda x: x.rolling(window=w, center=True).min())
                    rolled_list.append(val + '_rollingmin_' + str(w))

            for well in wells:
                features_fe.loc[features_fe.WELL == well, rolled_list] = \
                    features_fe.loc[features_fe.WELL == well, rolled_list].interpolate().bfill('rows').ffill('rows')
            features_fe[rolled_list].fillna(features_fe[rolled_list].median(), inplace=True)
            print("--- %s seconds ---" % (time.time() - start_time))

        if add_poly_features:
            features_fe['GR_DTC'] = features_fe['GR'] * features_fe['DTC']
            features_fe['GR_NPHI'] = features_fe['GR'] * features_fe['NPHI']

        if train:
            features_fe.to_csv('data_fe_train.zip', index=False, compression='gzip')

        return features_fe

    def pre_train(self, lithology_numbers, model_name, fe=False, train_all=False, add_seq_features=True,
                  add_poly_features=False, add_log_transformation=False):
        data = pd.read_csv('train.csv', sep=';')
        data = data.drop(columns=['SGR'])

        if fe:
            X = data[list(data.columns[:-2]) + ['FORCE_2020_LITHOFACIES_CONFIDENCE']].copy()
            X = self._preprocess(X, train=True, add_seq_features=add_seq_features, add_poly_features=add_poly_features,
                                 add_log_transformation=add_log_transformation)
        else:
            X = pd.read_csv('data_fe_train.zip', compression='gzip')

        y = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].copy()

        print(X.shape, y.shape)
        y = y.map(lithology_numbers)
        category_to_lithology = {y: x for x, y in lithology_numbers.items()}
        model = model_name.split('.')[0]

        clusterlist = pd.DataFrame(columns=['WELL', 'X_LOC', 'Y_LOC'])
        for i in X.WELL.unique():
            clusterlist = clusterlist.append(
                X[(X['WELL'] == i) & (X['X_LOC'] > 0) & (X['Y_LOC'] > 0)].iloc[0][['WELL', 'X_LOC', 'Y_LOC']])
        clusterlist = clusterlist.reset_index(drop=True)
        gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', random_state=2).fit(
            clusterlist[['X_LOC', 'Y_LOC']])
        joblib.dump(gmm, 'cluster_model.joblib.gz', compress=('gzip', 3))

        labels = gmm.predict(clusterlist[['X_LOC', 'Y_LOC']]) + 1
        clusterlist['cluster'] = labels

        score_list = []
        A = np.load('penalty_matrix.npy')

        for i in range(1, 4):
            well_names = clusterlist[clusterlist['cluster'] == i].WELL
            X_tmp = X[X.WELL.isin(well_names)]
            y_tmp = y[X_tmp.index]
            X_tmp = X_tmp.drop(columns=['FORCE_2020_LITHOFACIES_CONFIDENCE', 'WELL'])
            if i == 1:
                X_tmp = X_tmp.drop(columns=['MUDWEIGHT', 'DCAL'])
            elif i == 2:
                X_tmp = X_tmp.drop(columns=['RMIC'])
            elif i == 3:
                X_tmp = X_tmp.drop(columns=['MUDWEIGHT', 'DCAL', 'RMIC'])

            if model == 'xgboost':
                X_tmp = X_tmp.drop(columns=['GROUP', 'FORMATION', 'BS'])

                if train_all:
                    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_tmp)
                    print('Training cluster {} using {} with input shape {} {}'.format(i, model, X_tmp.shape, '-' * 20))

                    clf = XGBClassifier(tree_method='gpu_hist', learning_rate=0.12,
                                        max_depth=10, min_child_weight=10, n_estimators=1000,
                                        seed=43, eval_metric='auc', objective="multi:softmax",
                                        colsample_bytree=0.9).fit(X_tmp, y_tmp, sample_weight=weights, verbose=False)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.2,
                                                                        random_state=42, stratify=y_tmp)
                    print(
                        'Training cluster {} using {} with input shape {} {}'.format(i, model, X_train.shape, '-' * 20))

                    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
                    clf = XGBClassifier(tree_method='gpu_hist', learning_rate=0.12,
                                        max_depth=10, min_child_weight=10, n_estimators=1000,
                                        seed=43, eval_metric='auc', objective="multi:softmax",
                                        colsample_bytree=0.9).fit(X_train, y_train, sample_weight=weights, verbose=True)
                    y_pred = clf.predict(X_test)
                    score = get_score(A, y_test.values, y_pred)
                    score_list.append(score)
                    print("\n score: {}".format(score))
                    print(classification_report(y_test, y_pred))
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report = pd.DataFrame(report).transpose()
                    report.to_csv(model + '_classification_report' + str(i) + '.csv')
                    plot_confusion_matrix(y_true=np.vectorize(category_to_lithology.get)(y_test),
                                          y_pred=np.vectorize(category_to_lithology.get)(y_pred),
                                          classes=list(category_to_lithology.values()),
                                          title=model + '_confusion_matrix_' + str(i))
                ax = plot_importance(clf, title=model + '_feature_importance_' + str(i), max_num_features=15)
                ax.figure.tight_layout()
                ax.figure.savefig(model + '_feature_importance_' + str(i) + '.png')

            elif model == 'catboost':
                if i == 1:
                    X_tmp = X_tmp.drop(columns=['GR_rollingmean_30', 'GR_rollingmax_30', 'GR_rollingmin_30',
                                                'DTC_rollingmean_30', 'DTC_rollingmax_30', 'DTC_rollingmin_30'])

                best = {'eval_metric': 'TotalF1', 'depth': 10, 'l2_leaf_reg': 5, 'learning_rate': 0.15,
                        'loss_function': 'MultiClass', 'od_type': 'Iter', 'early_stopping_rounds': 100,
                        'task_type': 'GPU', 'verbose': 500, 'random_seed': 42}

                if i == 1:
                    best = {'eval_metric': 'TotalF1:use_weights=False', 'depth': 11, 'l2_leaf_reg': 2,
                            'learning_rate': 0.11679185273824155, 'fold_len_multiplier': 1.669990271921796,
                            'loss_function': 'MultiClass', 'od_type': 'Iter', 'od_wait': 25,
                            'task_type': 'GPU', 'verbose': 500, 'random_seed': 42}
                if i == 2:
                    best = {'eval_metric': 'TotalF1:use_weights=False', 'depth': 11, 'l2_leaf_reg': 7,
                            'learning_rate': 0.13207049403753848, 'fold_len_multiplier': 2.0345037506716057,
                            'loss_function': 'MultiClass', 'od_type': 'Iter', 'od_wait': 25, 'one_hot_max_size': 7,
                            'task_type': 'GPU', 'verbose': 500, 'random_seed': 42}
                # if i == 3:
                #     best = {'eval_metric': 'TotalF1:use_weights=False', 'depth': 12, 'l2_leaf_reg': 3,
                #             'learning_rate': 0.20557260217142548, 'fold_len_multiplier': 1.9159673725003985,
                #             'loss_function': 'MultiClass', 'od_type': 'Iter', 'od_wait': 25, 'one_hot_max_size': 11,
                #             'task_type': 'GPU', 'verbose': 500, 'random_seed': 42}

                cat_features = [s for s in X_tmp.columns if ('GROUP' in s) or ('BS' in s) or ('FORMATION' in s)]
                X_tmp[cat_features] = X_tmp[cat_features].astype(str)
                cat_features_index = [X_tmp.columns.get_loc(c) for c in cat_features if c in X_tmp]

                if train_all:
                    class_names = y_tmp.unique()
                    print('Training cluster {} using {} with input shape {} {}'.format(i, model, X_tmp.shape, '-' * 20))

                    weights = class_weight.compute_class_weight(class_weight='balanced', classes=class_names, y=y_tmp)
                    best['cat_features'] = cat_features_index
                    best['class_weights'] = weights
                    clf = CatBoostClassifier(**best).fit(X_tmp, y_tmp)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_tmp, y_tmp, test_size=0.2,
                                                                        random_state=42, stratify=y_tmp)
                    class_names = y_train.unique()
                    print(
                        'Training cluster {} using {} with input shape {} {}'.format(i, model, X_train.shape, '-' * 20))

                    weights = class_weight.compute_class_weight(class_weight='balanced', classes=class_names, y=y_train)
                    best['cat_features'] = cat_features_index
                    best['class_weights'] = weights
                    clf = CatBoostClassifier(**best).fit(X_train, y_train)

                    y_pred = clf.predict(X_test)

                    tmp = X_test.join(pd.DataFrame(np.vectorize(category_to_lithology.get)(y_pred),
                                                   columns=['prediction'], index=X_test.index))
                    tmp = tmp.join(pd.DataFrame(np.vectorize(category_to_lithology.get)(y_test),
                                                columns=['FORCE_2020_LITHOFACIES_LITHOLOGY'], index=X_test.index))
                    tmp.to_csv('test_pred_' + str(i) + '.csv')

                    score = get_score(A, y_test.values, y_pred)
                    score_list.append(score)
                    print("\n score: {}".format(score))
                    print(classification_report(y_test, y_pred))
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report = pd.DataFrame(report).transpose()
                    report.to_csv(model + '_classification_report' + str(i) + '.csv')
                    plot_confusion_matrix(y_true=np.vectorize(category_to_lithology.get)(y_test),
                                          y_pred=np.vectorize(category_to_lithology.get)(y_pred),
                                          classes=list(category_to_lithology.values()),
                                          title=model + '_confusion_matrix_' + str(i))
                plot_feature_importance(clf, title=model + '_feature_importance_' + str(i))

            joblib.dump(clf, model + '_' + str(i) + '.joblib.gz', compress=('gzip', 3))
            np.savetxt("scorelist.txt", score_list, delimiter=",", fmt='%s')

    def predict(self, features, clf_weight, add_seq_features=True, add_poly_features=False,
                add_log_transformation=False):

        X = self._preprocess(features, train=False, add_seq_features=add_seq_features,
                             add_poly_features=add_poly_features,
                             add_log_transformation=add_log_transformation)
        clusterlist = pd.DataFrame(columns=['WELL', 'X_LOC', 'Y_LOC'])
        for i in X.WELL.unique():
            clusterlist = clusterlist.append(
                X[(X['WELL'] == i) & (X['X_LOC'] > 0) & (X['Y_LOC'] > 0)].iloc[0][['WELL', 'X_LOC', 'Y_LOC']])
        clusterlist = clusterlist.reset_index(drop=True)
        gmm = joblib.load('cluster_model.joblib.gz')
        labels = gmm.predict(clusterlist[['X_LOC', 'Y_LOC']]) + 1
        clusterlist['cluster'] = labels

        pred = np.ones(len(X))

        for i in range(1, 4):
            print('Predicting cluster {}'.format(i))
            well_names = clusterlist[clusterlist['cluster'] == i].WELL
            X_test = X[X.WELL.isin(well_names)]
            X_test = X_test.drop(columns=['WELL'])

            if i == 1:
                X_test = X_test.drop(columns=['MUDWEIGHT', 'DCAL'])
            elif i == 2:
                X_test = X_test.drop(columns=['RMIC'])
            elif i == 3:
                X_test = X_test.drop(columns=['MUDWEIGHT', 'DCAL', 'RMIC'])

            X_test_xgb = X_test.copy()
            X_test_cat = X_test.copy()
            X_test_xgb = X_test_xgb.drop(columns=['GROUP', 'FORMATION', 'BS'])

            for version, file in enumerate(glob('xgb_models_' + str(i) + '*' + ".joblib.gz")):
                xgb_list = joblib.load(file)
                if version == 0:
                    y_pred_xgb = np.asarray([clf.predict_proba(X_test_xgb) for clf in xgb_list])
                else:
                    y_pred_xgb = np.concatenate(
                        [y_pred_xgb, np.asarray([clf.predict_proba(X_test_xgb) for clf in xgb_list])])

            for version, file in enumerate(glob('cat_tuned_models_' + str(i) + '*' + ".joblib.gz")):
                cat_tuned_list = joblib.load(file)
                if version == 0:
                    y_pred_cat_tuned = np.asarray([clf.predict_proba(X_test_cat) for clf in cat_tuned_list])
                else:
                    y_pred_cat_tuned = np.concatenate(
                        [y_pred_cat_tuned, np.asarray([clf.predict_proba(X_test_cat) for clf in cat_tuned_list])])

            print('Soft voting with classfier weights: {}'.format(clf_weight))
            # y_pred = np.concatenate([y_pred_xgb, y_pred_cat_normal, y_pred_cat_tuned])
            y_pred = np.concatenate([y_pred_xgb, y_pred_cat_tuned])
            y_pred = np.average(y_pred, axis=0, weights=clf_weight)
            y_pred = np.argmax(y_pred, axis=1)
            label_encoder = joblib.load('label_encoder_' + str(i) + '.joblib.gz')
            y_pred = label_encoder.inverse_transform(y_pred)
            pred[X_test_xgb.index] = np.array(y_pred)

        return pred

    def train(self, seed_xgb, seed_tuned_cat,
              lithology_numbers, fe=False, add_seq_features=True, add_poly_features=False,
              add_log_transformation=False):

        data = pd.read_csv('train.csv', sep=';')
        data = data.drop(columns=['SGR'])
        if fe:
            X = data[list(data.columns[:-2]) + ['FORCE_2020_LITHOFACIES_CONFIDENCE']].copy()
            X = self._preprocess(X, train=True, add_seq_features=add_seq_features, add_poly_features=add_poly_features,
                                 add_log_transformation=add_log_transformation)
        else:
            X = pd.read_csv('data_fe_train.zip', compression='gzip')
        y = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].copy()
        y = y.map(lithology_numbers)

        clusterlist = pd.DataFrame(columns=['WELL', 'X_LOC', 'Y_LOC'])
        for i in X.WELL.unique():
            clusterlist = clusterlist.append(
                X[(X['WELL'] == i) & (X['X_LOC'] > 0) & (X['Y_LOC'] > 0)].iloc[0][['WELL', 'X_LOC', 'Y_LOC']])
        clusterlist = clusterlist.reset_index(drop=True)
        gmm = mixture.GaussianMixture(n_components=3, covariance_type='full', random_state=2).fit(
            clusterlist[['X_LOC', 'Y_LOC']])
        joblib.dump(gmm, 'cluster_model.joblib.gz', compress=('gzip', 3))
        labels = gmm.predict(clusterlist[['X_LOC', 'Y_LOC']]) + 1
        clusterlist['cluster'] = labels

        for i in range(1, 4):
            print('Training cluster {}'.format(i))

            cat_tuned_list = []
            xgb_list = []

            well_names = clusterlist[clusterlist['cluster'] == i].WELL
            X_train = X[X.WELL.isin(well_names)]
            X_train = X_train.drop(columns=['FORCE_2020_LITHOFACIES_CONFIDENCE', 'WELL'])
            y_train = y[X_train.index]

            label_encoder = LabelEncoder()
            label_encoder.fit(y_train)
            joblib.dump(label_encoder, 'label_encoder_' + str(i) + '.joblib.gz', compress=('gzip', 3))

            if i == 1:
                X_train = X_train.drop(columns=['MUDWEIGHT', 'DCAL'])
                best_xgb = {'colsample_bytree': 0.6457635688501269, 'gamma': 1,
                            'lambda': 0.9, 'learning_rate': 0.20408792946960846, 'max_depth': 15,
                            'min_child_weight': 4, 'subsample': 0.7,
                            'tree_method': 'gpu_hist', 'eval_metric': 'mlogloss', 'objective': 'multi:softmax',
                            'seed': 42}

                best_tuned_cat = {'eval_metric': 'TotalF1:use_weights=False', 'depth': 11, 'l2_leaf_reg': 2,
                                  'learning_rate': 0.11679185273824155, 'fold_len_multiplier': 1.669990271921796,
                                  'loss_function': 'MultiClass', 'od_type': 'Iter', 'od_wait': 25,
                                  'task_type': 'GPU', 'verbose': 500, 'random_seed': 42}

            elif i == 2:
                X_train = X_train.drop(columns=['RMIC'])

                best_xgb = {'colsample_bytree': 0.9231871602710936, 'eval_metric': 'mlogloss', 'gamma': 1,
                            'lambda': 0.9,
                            'learning_rate': 0.21381650119853535, 'max_depth': 16, 'min_child_weight': 5,
                            'num_class': 12,
                            'objective': 'multi:softmax', 'subsample': 0.7, 'tree_method': 'gpu_hist', 'seed': 42}

                best_tuned_cat = {'custom_metric': 'TotalF1:use_weights=True', 'depth': 12,
                                  'fold_len_multiplier': 1.4597904591216981,
                                  'l2_leaf_reg': 4, 'learning_rate': 0.2154607024950838, 'loss_function': 'MultiClass',
                                  'od_type': 'Iter',
                                  'od_wait': 25, 'one_hot_max_size': 7, 'task_type': 'GPU', 'verbose': 500,
                                  'random_seed': 42}


            elif i == 3:
                X_train = X_train.drop(columns=['MUDWEIGHT', 'DCAL', 'RMIC'])

                best_xgb = {'colsample_bytree': 0.986555257438249, 'eval_metric': 'mlogloss', 'gamma': 1,
                            'lambda': 0.93,
                            'learning_rate': 0.12228232369801341, 'max_depth': 19, 'min_child_weight': 4,
                            'num_class': 9,
                            'objective': 'multi:softmax', 'subsample': 0.7, 'tree_method': 'gpu_hist', 'seed': 42}

                best_tuned_cat = {'custom_metric': 'TotalF1:use_weights=True', 'depth': 12,
                                  'fold_len_multiplier': 1.0351256868651992, 'l2_leaf_reg': 3,
                                  'learning_rate': 0.24476582315993148, 'loss_function': 'MultiClass',
                                  'od_type': 'Iter', 'od_wait': 25, 'one_hot_max_size': 9,
                                  'task_type': 'GPU', 'verbose': 500, 'random_seed': 42}

            X_train_xgb = X_train.copy()
            X_train_cat = X_train.copy()
            X_train_xgb = X_train_xgb.drop(columns=['GROUP', 'FORMATION', 'BS'])

            sample_weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
            cat_features = [s for s in X_train_cat.columns if ('GROUP' in s) or ('BS' in s) or ('FORMATION' in s)]
            X_train_cat[cat_features] = X_train_cat[cat_features].astype(str)
            cat_features_index = [X_train_cat.columns.get_loc(c) for c in cat_features if c in X_train_cat]
            class_names = y_train.unique()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=class_names, y=y_train)

            version = 1
            for j in range(1, seed_xgb + 1):
                start_time = time.time()
                best_xgb['seed'] = j + 1000000
                best_xgb['num_class'] = len(y_train.unique())
                clf_xgb = XGBClassifier(**best_xgb).fit(X_train_xgb, y_train, sample_weight=sample_weights)
                xgb_list.append(clf_xgb)
                print('Training seed {} using xgboost with input shape {}, it took {:05.2f} seconds {}'.format(
                    j, X_train_xgb.shape, (time.time() - start_time), '-' * 20))
                if j % 5 == 0:
                    joblib.dump(xgb_list, "xgb_models_" + str(i) + "_" + str(version) + ".joblib.gz",
                                compress=('gzip', 5))
                    version = version + 1
                    xgb_list = []

            version = 1
            for p in range(1, seed_tuned_cat + 1):
                start_time = time.time()
                best_tuned_cat['cat_features'] = cat_features_index
                best_tuned_cat['class_weights'] = class_weights
                best_tuned_cat['random_seed'] = 100000000 + p
                clf_cat = CatBoostClassifier(**best_tuned_cat).fit(X_train_cat, y_train,
                                                                   cat_features=cat_features_index)
                cat_tuned_list.append(clf_cat)
                print('Training seed {} using tuned catboost with input shape {}, it took {:05.2f} seconds {}'.format(
                    1000000 + p, X_train_cat.shape, (time.time() - start_time), '-' * 20))
                if p % 5 == 0:
                    joblib.dump(cat_tuned_list, "cat_tuned_models_" + str(i) + "_" + str(version) + ".joblib.gz",
                                compress=('gzip', 5))
                    version = version + 1
                    cat_tuned_list = []


if __name__ == '__main__':
    train = False
    test = True

    feature_engineering = False
    add_seq_features = True
    add_poly_features = True
    add_log_transformation = True

    seed_xgb = 20
    seed_tuned_cat = 20
    clf_weight = [1] * seed_xgb + [2.2] * seed_tuned_cat

    lithology_numbers = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 70032: 6, 88000: 7, 86000: 8,
                         99000: 9, 90000: 10, 93000: 11}

    model = Model()
    if train:
        model.train(seed_xgb=seed_xgb, seed_tuned_cat=seed_tuned_cat,
                    lithology_numbers=lithology_numbers, fe=feature_engineering,
                    add_seq_features=add_seq_features, add_poly_features=add_poly_features,
                    add_log_transformation=add_log_transformation)
    if test:
        open_test_features = pd.read_csv('test.csv', sep=';')
        open_test_features = open_test_features.drop(columns=['SGR'])

        test_prediction = model.predict(open_test_features, clf_weight,
                                        add_seq_features=add_seq_features, add_poly_features=add_poly_features,
                                        add_log_transformation=add_log_transformation)

        category_to_lithology = {y: x for x, y in lithology_numbers.items()}
        test_prediction_for_submission = np.vectorize(category_to_lithology.get)(test_prediction)
        print(test_prediction_for_submission)
        np.savetxt('test_predictions.csv', test_prediction_for_submission, header='lithology', fmt='%i', comments='')