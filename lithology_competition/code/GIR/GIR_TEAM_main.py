import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBClassifier

import pywt
import pickle
# import missingno as mno
import precond
import wavelet_transform
import feature_augmentation
import validation
import imputation

from datetime import datetime
start_time = datetime.now()

# DATASET IMPORTATION
train = pd.read_csv('train.csv', sep=';')
train_bkp = train.copy()

# MISSING VALUES REPRESENTATION
# mno.matrix(train, figsize = (20, 6))

# DESCRIPTIVE STATISTICS
train_stats = train.describe()

# PREPROCESSING
# train = precond.precond_train(train)

# LABEL ENCODER
# le = LabelEncoder()
# le.fit(train['GROUP'])
# train['GROUP'] = le.transform(train['GROUP'])

# save label encoder model
# pickle.dump(le, open('labelencoder.pkl', 'wb'))

# REMOVING UNWANTED COLUMNS FOR NOW
# train_imp = train.drop(['WELL','FORCE_2020_LITHOFACIES_LITHOLOGY','FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)

# IMPUTATION FOR MISSING VALUES
# train_imp, model_imp_list = imputation.imputer_train(train_imp, "train_imp.csv")

# ADDING WELL COLUMN BACK TO DATAFRAME
# train_imp['WELL'] = train['WELL']

# FEATURE SELECTION
features = ['WELL','X_LOC','Y_LOC','Z_LOC','RDEP','GROUP','CALI','GR','RHOB','NPHI','PEF','DTC','SP','DRHO']

# train_imp = train_imp[features]

# ADDING FACIES COLUMN BACK TO DATAFRAME
# train_imp['FORCE_2020_LITHOFACIES_LITHOLOGY'] = train['FORCE_2020_LITHOFACIES_LITHOLOGY']

# WAVELET TRANSFORM
# train_imp = wavelet_transform.wavelet_transform(train_imp)

# SHOULDER EFFECT REMOVAL
# train_imp.loc[train_imp['FORCE_2020_LITHOFACIES_LITHOLOGY'].shift(-1) != train_imp['FORCE_2020_LITHOFACIES_LITHOLOGY'].shift(1), 'FORCE_2020_LITHOFACIES_LITHOLOGY'] = np.nan
# train_imp.dropna(subset=['FORCE_2020_LITHOFACIES_LITHOLOGY'], axis=0, inplace=True)

# FEATURE AUGMENTATION
# train_imp_aug = train_imp.drop(['WELL','FORCE_2020_LITHOFACIES_LITHOLOGY'], axis=1)
# train_imp_aug = feature_augmentation.feat_aug(train_imp_aug, train_imp['WELL'], train_imp['Z_LOC'])
# train_imp_aug = feature_augmentation.poly_feat(train_imp_aug)
  
# TRAINING DATA AND LABEL
# X = train_imp_aug
# y = train_imp['FORCE_2020_LITHOFACIES_LITHOLOGY']

# TRAIN AND TEST SPLIT
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=0)

# MODEL SELECTION
# model = XGBClassifier(max_depth=3,
#                       learning_rate=0.1,
#                       n_estimators=300,
#                       random_state=0)
# model.fit(X_train, y_train)
# print("Model trained")

# TEST SPLIT PREDICTION AND METRICS PERFORMANCE
# accuracy_split_test, pen_rel_split_test = validation.validation(X_test, y_test, train_imp, model, 'penalty_matrix.npy') 

# SAVE PRE-TRAINED MODEL
# pickle.dump(model, open('model.pkl', 'wb'))

##############################################################################

# CLOSED TEST DATA IMPORTATION
test = pd.read_csv('test.csv', sep=';')     # put here the closed dataset for prediction
test_bkp = test.copy()

# PREPROCESSING
test = precond.precond_test(test)

# LABEL ENCODER
le = pickle.load(open('labelencoder.pkl', 'rb'))
test['GROUP'] = le.transform(test['GROUP'])

# REMOVING UNWANTED COLUMNS FOR NOW
test_imp = test.drop(['WELL'], axis=1)

# IMPUTATION FOR MISSING VALUES
train_imp_csv = pd.read_csv('train_imp.csv', sep=';')
test_imp = imputation.imputer_test(test_imp, train_imp_csv)

# ADDING WELL COLUMN BACK TO DATAFRAME
test_imp['WELL'] = test['WELL']

# FEATURE SELECTION
test_imp = test_imp[features]

# WAVELET TRANSFORM
test_imp = wavelet_transform.wavelet_transform(test_imp)

# FEATURE AUGMENTATION
test_imp_aug = test_imp.drop(['WELL'], axis=1)
test_imp_aug = feature_augmentation.feat_aug(test_imp_aug, test_imp['WELL'], test_imp['Z_LOC'])
test_imp_aug = feature_augmentation.poly_feat(test_imp_aug)

# PREDICTION ON CLOSED TEST DATA
model = pickle.load(open('model.pkl', 'rb'))

prediction = model.predict(test_imp_aug)

# FACIES PREDICTION REFINEMENT
test['PREDICTION'] = prediction
test.loc[test['PREDICTION'].shift(-2) == test['PREDICTION'].shift(1), 'PREDICTION'] = test['PREDICTION'].shift(-2)
test.loc[test['PREDICTION'].shift(-1) == test['PREDICTION'].shift(1), 'PREDICTION'] = test['PREDICTION'].shift(-1)
test['PREDICTION'].iloc[-1] = prediction[-1]

# WRITTING A CSV FILE
test_prediction_for_submission = test['PREDICTION'].values
np.savetxt('GIR_TEAM_final_submission.csv', test_prediction_for_submission, header='lithology', comments='', fmt='%i')

# DURATION TIME
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))