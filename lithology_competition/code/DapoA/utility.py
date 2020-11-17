import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RidgeClassifier


def data_modify(df):
  for col in ['RSHA', 'RMED', 'RDEP']:
    idx=df[df[col]<0].index
    df[col][df.index.isin(idx)]=np.nan
    idx=df[df[col]>2000].index
    df[col][df.index.isin(idx)]=2000
    df[col]=np.log10(df[col])  #convert all restivity logs to logarithmic scale
  for col in ['ROP']:
    idx=df[df[col]<=0].index
    df[col][df.index.isin(idx)]=np.nan
    df[col]=np.log10(df[col])   #convert all log to logarithmic scale
  for col in ['GR']:
    idx=df[df[col]>200].index
    df[col][df.index.isin(idx)]=200
  return df

#impute missing categorical values
def impute_nan(df, ds, dF):
    if ds.isnull().any()==True:
        labeler_st = LabelEncoder()
        rc_st = RidgeClassifier(tol=1e-2, solver="sag")
        Sg = Series(labeler_st.fit_transform(ds.astype(str)), index=ds.index)
        Sg = Sg.where(ds.notnull(), ds, axis=0)
        x_notna = df.GR[Sg.notnull()].to_numpy().reshape(-1, 1)
        y_notna = Sg[Sg.notnull()].to_numpy().astype('int').ravel()
        x_nan = df.GR[Sg.isnull()].to_numpy().reshape(-1, 1)
        rc_st.fit(x_notna,y_notna)
        Sg[Sg.isnull()]=rc_st.predict(x_nan)
        Sg=Series(Sg, index=ds.index).astype(int)
        ds=Series(labeler_st.inverse_transform(Sg.values.ravel()), index=ds.index)
        #print('\nStratigraphy:', np.unique(ds))
    if dF.isnull().any()==True:
        rc_fm = RidgeClassifier(tol=1e-2, solver="sag")
        labeler_fm = LabelEncoder()
        Fm = Series(labeler_fm.fit_transform(dF.astype(str)), index=dF.index)
        labeler_st = LabelEncoder()
        Sg=Series(labeler_st.fit_transform(ds.astype(str)), index=ds.index)
        Fm=Fm.where(dF.notnull(), dF, axis=0)
        x_notna = np.concatenate((df.GR[Fm.notnull()].to_numpy().reshape(-1, 1), 
                                  Sg[Fm.notnull()].to_numpy().reshape(-1, 1)), 
                                 axis=1)
        y_notna = Fm[Fm.notnull()].to_numpy().astype('int').ravel()
        x_nan = np.concatenate((df.GR[Fm.isnull()].to_numpy().reshape(-1, 1), 
                                Sg[Fm.isnull()].to_numpy().reshape(-1, 1)), axis=1)
        rc_fm.fit(x_notna,y_notna)
        Fm[Fm.isnull()]=rc_fm.predict(x_nan)
        Fm=Series(Fm, index=dF.index).astype(int)
        dF=Series(labeler_fm.inverse_transform(Fm.values.ravel()), index=dF.index)
        #print('\nFormation:', np.unique(dF))
    return Sg, Fm

def combine_features(df, formation, strat):
    df=pd.concat([df, formation, strat], axis=1).rename(columns = {0:'Formation', 1:'Strata'})
    
    return df

# Feature augmentation function
def augment_features(X, well, depth):
    
    # Augment features
    padded_rows = []
    X_aug = np.zeros((X.shape[0], X.shape[1]*2))
    for w in np.unique(well):
        w_idx = np.where(well == w)[0]  #index
        
        # Compute features gradient function
        d_diff = np.diff(depth[w_idx]).reshape((-1, 1)) #check difference in depth
        d_diff[d_diff==0] = 0.01                        #if difference in depth is zero, make it 0.01
        X_diff = np.diff(X.loc[w_idx, :], axis=0)           #check difference in features
        X_grad = X_diff / d_diff                        #calculate gradient by dividing features by depth 
        
        # Compensate for last rows - padding
        X_grad = np.concatenate((X_grad, np.zeros((1, X_grad.shape[1]))))
    
        X_aug[w_idx, :] = np.concatenate((X.loc[w_idx, :], X_grad), axis=1)
        
        # Find padded rows
        padded_rows.append(w_idx[-1])
                    
    return X_aug