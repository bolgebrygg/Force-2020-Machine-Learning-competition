import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer # LEMBRARRR
from sklearn.impute import IterativeImputer
#from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sns
#import missingno as mno

def imputer_train(data, output_name):

    # FEATURES CORRELATION
    plt.figure(figsize=(12,10))
    corr_data = data.corr()
    sns.heatmap(corr_data, annot=True, cmap=plt.cm.Reds)
    plt.title('Features correlation train')
    plt.savefig('Features correlation')

    target_and_feat = {
        'DEPTH_MD': ['Z_LOC'],
        'X_LOC': ['Y_LOC'], 
        'Y_LOC': ['X_LOC'], 
        'Z_LOC': ['DEPTH_MD'],
        'GROUP': ['DTC', 'RHOB' ,'RDEP', 'RSHA'], 
        'CALI': ['BS', 'DTC', 'RHOB', 'Z_LOC', 'NPHI'],
        'RSHA': ['RXO', 'RDEP', 'RMED', 'RMIC'], 
        'RMED': ['RSHA', 'RDEP', 'DRHO', 'DTS'], 
        'RDEP': ['RSHA', 'RMED', 'NPHI', 'RXO'], 
        'RHOB': ['DTC', 'Z_LOC', 'DTS', 'NPHI', 'CALI'], 
        'GR': ['SGR'], 
        'SGR': ['ROPA', 'GR'],
        'NPHI': ['DTS', 'DTC', 'RHOB', 'Z_LOC'], 
        'PEF': ['DCAL', 'MUDWEIGHT', 'GR'], 
        'DTC': ['DTS', 'RHOB', 'NPHI', 'Z_LOC'], 
        'SP': ['ROPA','NPHI', 'GR'],
        'BS': ['CALI', 'Z_LOC', 'DTC'], 
        'ROP': ['ROPA', 'DTS'],
        'DTS': ['DTC', 'RHOB', 'NPHI', 'Z_LOC'],
        'DCAL': ['RHOB', 'CALI'],
        'DRHO': ['DCAL', 'RMED', 'SGR'], 
        'MUDWEIGHT': ['CALI', 'PEF'], 
        'RMIC': ['RSHA', 'DTS', 'RXO'], 
        'ROPA': ['Z_LOC', 'RHOB', 'DTC', 'SP', 'ROP'],
        'RXO': ['RMIC', 'DTS', 'RDEP', 'RSHA']}

    # SEPARATING FEATURES WITH HIGH CORRELATION
    imp_data_list = []
    imp_model_list = []

    for feature in corr_data:
        
        #Correlation with output variable
        corr_target = abs(corr_data[feature])
        corr_target.drop(feature, inplace=True)
    
        #Selecting highly correlated features     
        feat = target_and_feat.get(feature)
        relevant_features = corr_target[feat]
            
       
        data_df = pd.concat([data[relevant_features.index], data[feature]], axis=1)
        
        # Samples where all features values are nan
        try:
            a = data_df[data_df.columns[0]].isna()
            b = data_df[data_df.columns[1]].isna()
            c = data_df[data_df.columns[2]].isna()        
            d = data_df[data_df.columns[3]].isna() 
            e = data_df[data_df.columns[4]].isna() 
        except: 
            pass
        
        if data_df.shape[1] == 5:
            condition = a & b & c & d & e
        elif data_df.shape[1] == 4:
            condition = a & b & c & d
        elif data_df.shape[1] == 3:
            condition = a & b & c
        elif data_df.shape[1] == 2:
            condition = a & b
        else:
            condition = a
        index_nan_data = data_df[condition].index
     
        data_df2 = data_df.copy()
        data_df2[feature] = np.nan
        
        # IMPUTATION FOR MISSING VALUES
        imp_model = IterativeImputer(estimator=ExtraTreesRegressor(max_depth=15), random_state=0)
        imp_model.fit(data_df)
        data_imp = pd.DataFrame(imp_model.transform(data_df2), columns=data_df2.columns, index=data.index)

        imp_model_list.append(imp_model)

        data_imp[feature] = data_df[feature].combine_first(data_imp[feature])
        
        data_imp[feature][index_nan_data] = np.nan
         
        imp_data_list.append(data_imp.iloc[:,-1])
        
        print("Imputation done", feature)


    data_imp = pd.concat(imp_data_list, axis=1)

    imp_model2 = IterativeImputer(estimator=ExtraTreesRegressor(max_depth=10), random_state=0)
    imp_model2.fit(data_imp)
    data_imp = pd.DataFrame(imp_model2.transform(data_imp), columns=data_imp.columns, index=data.index)
    
    imp_model_list.append(imp_model2)
    
    # SAVING AS NEW CSV FILES
    data_imp.to_csv(output_name, index=False, sep=';')
    
    return data_imp, imp_model_list
    

def imputer_test(data, train_csv):
        
    # FEATURES CORRELATION
    plt.figure(figsize=(12,10))
    corr_data = data.corr()
    sns.heatmap(corr_data, annot=True, cmap=plt.cm.Reds)
    plt.title('Features correlation test')
    plt.savefig('Features correlation')

    target_and_feat = {
        'DEPTH_MD': ['Z_LOC'],
        'X_LOC': ['Y_LOC'], 
        'Y_LOC': ['X_LOC'], 
        'Z_LOC': ['DEPTH_MD'],
        'GROUP': ['DTC', 'RHOB' ,'RDEP', 'RSHA'], 
        'CALI': ['BS', 'DTC', 'RHOB', 'Z_LOC', 'NPHI'],
        'RSHA': ['RXO', 'RDEP', 'RMED', 'RMIC'], 
        'RMED': ['RSHA', 'RDEP', 'DRHO', 'DTS'], 
        'RDEP': ['RSHA', 'RMED', 'NPHI', 'RXO'], 
        'RHOB': ['DTC', 'Z_LOC', 'DTS', 'NPHI', 'CALI'], 
        'GR': ['SGR'], 
        'SGR': ['ROPA', 'GR'],
        'NPHI': ['DTS', 'DTC', 'RHOB', 'Z_LOC'], 
        'PEF': ['DCAL', 'MUDWEIGHT', 'GR'], 
        'DTC': ['DTS', 'RHOB', 'NPHI', 'Z_LOC'], 
        'SP': ['ROPA','NPHI', 'GR'],
        'BS': ['CALI', 'Z_LOC', 'DTC'], 
        'ROP': ['ROPA', 'DTS'],
        'DTS': ['DTC', 'RHOB', 'NPHI', 'Z_LOC'],
        'DCAL': ['RHOB', 'CALI'],
        'DRHO': ['DCAL', 'RMED', 'SGR'], 
        'MUDWEIGHT': ['CALI', 'PEF'], 
        'RMIC': ['RSHA', 'DTS', 'RXO'], 
        'ROPA': ['Z_LOC', 'RHOB', 'DTC', 'SP', 'ROP'],
        'RXO': ['RMIC', 'DTS', 'RDEP', 'RSHA']}

    # SEPARATING FEATURES WITH HIGH CORRELATION
    imp_data_list = []
    
    for feature in corr_data:
        
        #Correlation with output variable
        corr_target = abs(corr_data[feature])
        corr_target.drop(feature, inplace=True)

        #Selecting highly correlated features
        feat = target_and_feat.get(feature)
        relevant_features = corr_target[feat]
        
        train_df = pd.concat([train_csv[relevant_features.index], train_csv[feature]], axis=1)
        data_df = pd.concat([data[relevant_features.index], data[feature]], axis=1)
        
        # IMPUTATION FOR MISSING VALUES
        model = IterativeImputer(estimator=ExtraTreesRegressor(max_depth=15), random_state=0)
        model.fit(train_df) 
        data_imp = pd.DataFrame(model.transform(data_df), columns=data_df.columns, index=data.index)

        imp_data_list.append(data_imp.iloc[:,-1])
    
        print("Imputation done", feature)

    data_imp = pd.concat(imp_data_list, axis=1)
    
    return data_imp
