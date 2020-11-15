import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class DepthTransformer(TransformerMixin, BaseEstimator):
    """
    Measure features are augmented with gradients, derivatives and local means w.r.t. depth
    """

    _depth_feat = 'DEPTH_MD'
    _meas_feat = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS',
                  'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
    _well_feat = 'WELL'

    def fit(self, X: pd.DataFrame, y=None):

        if self._well_feat not in X:
            raise ValueError('Missing column: WELL')

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if self._well_feat not in X:
            raise ValueError('Missing column: WELL')

        Xout = X.copy()

        wells_groupby = X.groupby('WELL')

        for feat in self._meas_feat:
            if feat in X:
                feat_name = f'{feat}_grad'
                Xout[feat_name] = np.zeros(len(Xout), dtype=np.float32)
                feat_name = f'{feat}_der'
                Xout[feat_name] = np.zeros(len(Xout), dtype=np.float32)
                feat_name = f'{feat}_rm'
                Xout[feat_name] = np.zeros(len(Xout), dtype=np.float32)

        for well, well_idxs in wells_groupby.groups.items():

            Xwell = X.loc[well_idxs].sort_values('DEPTH_MD')
            depth_diff = np.ediff1d(Xwell[self._depth_feat], to_begin=1)
            for feat in self._meas_feat:
                if feat in X:
                    valid_idxs = Xwell[feat].notna()

                    # Gradient
                    feat_name = f'{feat}_grad'
                    if sum(valid_idxs) > 1:
                        grad = np.gradient(Xwell.loc[valid_idxs, feat], Xwell.loc[valid_idxs, self._depth_feat], axis=0)
                        grad[np.isnan(grad)] = 0
                        Xout.loc[valid_idxs[valid_idxs].index, feat_name] = grad

                    # Derivative
                    feat_name = f'{feat}_der'
                    if sum(valid_idxs) > 1:
                        der = np.ediff1d(Xwell.loc[valid_idxs, feat], to_begin=0) / depth_diff[valid_idxs]
                        der[np.isnan(der)] = 0
                        Xout.loc[valid_idxs[valid_idxs].index, feat_name] = der

                    # Local mean, 10m moving average
                    feat_name = f'{feat}_rm'
                    if sum(valid_idxs):
                        rm = Xwell.loc[valid_idxs, feat].rolling(min(67, sum(valid_idxs)), center=True).mean()
                        rm[np.isnan(rm)] = 0
                        Xout.loc[valid_idxs[valid_idxs].index, feat_name] = rm

        return Xout
