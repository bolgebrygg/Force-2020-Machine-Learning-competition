import pandas as pd
import scipy.signal
from sklearn.base import TransformerMixin, BaseEstimator


class LpfTransformer(TransformerMixin, BaseEstimator):
    """
    Measure features are lowpassed with a third order Butterworth filter at fc=0.2
    """

    _meas_feat = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS',
                  'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
    _well_feat = 'WELL'

    def __init__(self, fc: float = 0.2):
        """
        :param fc: Cut frequency
        """
        super().__init__()
        self.fc = fc
        self._b, self._a = scipy.signal.butter(3, fc)

    def fit(self, X: pd.DataFrame, y=None):

        if self._well_feat not in X:
            raise ValueError('Missing column: WELL')

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if self._well_feat not in X:
            raise ValueError('Missing column: WELL')

        Xout = X.copy()

        wells_groupby = X.groupby('WELL')

        for well, well_idxs in wells_groupby.groups.items():

            Xwell = X.loc[well_idxs].sort_values('DEPTH_MD')
            if len(Xwell) > 12:  # With less samples can't really filter
                for feat in self._meas_feat:
                    if feat in X:
                        feat_lpf = scipy.signal.filtfilt(self._b, self._a, Xwell[feat])
                        Xout.loc[well_idxs, feat] = feat_lpf

        return Xout
