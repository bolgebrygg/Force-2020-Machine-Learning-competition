import numpy as np
from sklearn.preprocessing import PolynomialFeatures

'''
This step was adapted from the ISPL Team submission in the SEG Geophysical Tutorial Machine Learning Contest 2016.
https://github.com/seg/2016-ml-contest/blob/master/ispl/facies_classification_try03_v2.ipynb
''' 

def feat_aug_gradient(data, depth):
    
    # Compute features gradient
    d_diff = np.diff(depth).reshape((-1, 1)) 
    d_diff[d_diff==0] = 0.001
    data_diff = np.diff(data, axis=0)
    data_grad = data_diff / d_diff

    data_grad = np.concatenate((data_grad, np.zeros((1, data_grad.shape[1]))))
    
    return data_grad

def feat_aug(data, well, depth):
    
    # Augment features
    data_aug = np.zeros((data.shape[0], data.shape[1]*2))
    for w in np.unique(well.astype(str)):
        data_aug_grad = feat_aug_gradient(data, depth)
        data_aug = np.concatenate((data, data_aug_grad), axis=1)
    
    return data_aug

def poly_feat(data_aug):
    deg = 2
    poly = PolynomialFeatures(deg, interaction_only=False)
    data_aug = poly.fit_transform(data_aug)
    data_aug = data_aug[:,1:]

    return data_aug