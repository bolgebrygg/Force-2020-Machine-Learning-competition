import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class LogTransformer(TransformerMixin, BaseEstimator):
    """
    Add Log of measures
    """

    _meas_feat = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS',
                  'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
    _well_feat = 'WELL'

    def fit(self, X: pd.DataFrame, y=None):

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        Xout = X.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for feat in self._meas_feat:
                if feat in X:
                    Xout[f'{feat}_log'] = np.log(Xout[feat])

        return Xout
