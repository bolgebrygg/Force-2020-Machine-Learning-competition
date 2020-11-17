import pandas as pd
import numpy as np

def precond_train(data):
    
    # REMOVING NAN VALUES FROM GROUP FEATURE
    data.dropna(subset=['GROUP'], axis=0, inplace=True)
    
    # MANIPULATING ANOMALOUS DATA
    data['ROPA'] = data['ROPA'].round(2).replace(-999.25, np.nan)
    data['RXO'][data['RXO'] < -10] = np.nan
    data['RXO'][data['RXO'] < 0] = 0
    data['RMED'][data['RMED'] < 0] = 0
    
    # REPLACING DCAL COLUMN FOR MORE RELIABLE DATA
    data['DCAL2'] = data['CALI'] - data['BS']
    data['DCAL'] = data['DCAL2'].combine_first(data['DCAL'])
    data = data.drop(['DCAL2'], axis=1)
    
    # X_LOC AND Y_LOC IMPUTATION
    data['X_LOC'] = data['X_LOC'].ffill()
    data['Y_LOC'] = data['Y_LOC'].ffill()

    # REMOVING ANOMALOUS VALUES
    index_sp = data[data['SP'] < -350].index
    indexDRHOmin = data[data['DRHO'] < -1.5].index
    indexDRHOmax = data[data['DRHO'] > 1.5].index
    indexDCALmax = data[data['DCAL'] > 8].index
    indexDCALmin = data[data['DCAL'] < -8].index
    
    data.drop(index_sp, inplace=True)
    data.drop(indexDRHOmin, inplace=True)
    data.drop(indexDRHOmax, inplace=True)
    data.drop(indexDCALmax, inplace=True)
    data.drop(indexDCALmin, inplace=True)
    
    return data

def precond_test(data):
    
    # MANIPULATING ANOMALOUS DATA
    data['ROPA'] = data['ROPA'].round(2).replace(-999.25, np.nan)
    data['RXO'][data['RXO'] < -10] = np.nan
    data['RXO'][data['RXO'] < 0] = 0
    data['RMED'][data['RMED'] < 0] = 0
    
    # REPLACING DCAL COLUMN FOR MORE RELIABLE DATA
    data['DCAL2'] = data['CALI'] - data['BS']
    data['DCAL'] = data['DCAL2'].combine_first(data['DCAL'])
    data = data.drop(['DCAL2'], axis=1)
    
    # X_LOC AND Y_LOC IMPUTATION
    data['X_LOC'] = data['X_LOC'].ffill()
    data['Y_LOC'] = data['Y_LOC'].ffill()
    
    return data
