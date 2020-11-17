# Script by Bohdan Pavlyshenko, b.pavlyshenko@gmail.com, https://www.linkedin.com/in/bpavlyshenko/
#
## Some code in this script was taken from the starter jupyter notebook from 
# competition 'FORCE: Machine Predicted Lithology' (https://xeek.ai/challenges/force-well-logs/overview)

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression

# Set up options 

# test file name
test_file_name='test.csv'

# prediction file 
pred_file='prediction.csv'

# model directory
model_dir='models/'

##########################

features_set1=['DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC', 'GROUP', 'FORMATION',
       'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR','SGR', 'NPHI', 'PEF',
       'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC',
       'ROPA', 'RXO']

features_set2=['Z_LOC', 'GROUP', 'FORMATION',
       'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF',
       'DTC', 'SP', 'BS', 'ROP', 'DTS', 'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC',
       'ROPA', 'RXO']
stack_list=['par1','par2','par3', 'par4','par5','par6', 'par7','par8']

data_test=pd.read_csv(test_file_name, sep=';')

enc=pickle.load(open(model_dir+'formation_enc.pkl', 'rb'))
data_test.FORMATION=data_test.FORMATION.astype(str)
unique_v_d=list(set(data_test.FORMATION.unique().tolist())-set(enc.classes_))
data_test.loc[data_test.FORMATION.isin(unique_v_d),'FORMATION']='nan'
data_test.FORMATION=enc.transform(data_test.FORMATION)

enc=pickle.load(open(model_dir+'group_enc.pkl', 'rb'))
data_test.GROUP=data_test.GROUP.astype(str)
unique_v_d=list(set(data_test.GROUP.unique().tolist())-set(enc.classes_))
data_test.loc[data_test.GROUP.isin(unique_v_d),'GROUP']='nan'
data_test.GROUP=enc.transform(data_test.GROUP)

stack_test_df=[]
for i in stack_list:
    print(i)
    if (i=='par8'):
        features=features_set2
    else:
        features=features_set1
    model=pickle.load(open(model_dir+'lgb_model_'+str(i)+'.pkl', 'rb'))
    tst_st=model.predict(data_test[features])
    stack_test_df.append(tst_st)

Xtest=np.hstack(stack_test_df)
ntest=Xtest.shape[0]
test_res_p=np.zeros([ntest,12])
for i in np.arange(12):
    print ('class',i)
    lr=pickle.load(open(model_dir+'lr_model_'+str(i)+'.pkl', 'rb'))
    test_res_p[:,i]=lr.predict_proba(Xtest)[:,1]
test_pred=np.argmax(test_res_p,axis=1)

# This part of the code is based on the code from the starter jupyter notebook 
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
category_to_lithology = {y:x for x,y in lithology_numbers.items()}
test_prediction_for_submission = np.vectorize(category_to_lithology.get)(test_pred)
np.savetxt(pred_file, test_prediction_for_submission, header='lithology', comments='', fmt='%i')
################################################
print("Done!")