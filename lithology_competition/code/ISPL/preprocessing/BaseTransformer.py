import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from utils import lithology


class BaseTransformer(TransformerMixin, BaseEstimator):
    """
    - One-hot encoding of categorial features (FORMATION,GROUP). Null samples are encoded as all-zeros.
    - Labels encoding according to the challenge score function
    - Confidence encoding from the 1,2,3 system of the challenge to 1,0.75,0.5
    - Pass-through real-valued features
    """

    _geom_feat = ['X_LOC', 'Y_LOC', 'Z_LOC', 'DEPTH_MD']
    _meas_feat = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS',
                  'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']
    _real_feat = _geom_feat + _meas_feat

    _cat_feat = ['FORMATION', 'GROUP']

    _label_feat = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
    _conf_feat = 'FORCE_2020_LITHOFACIES_CONFIDENCE'
    _well_feat = 'WELL'

    def __init__(self, *, keep_wells: bool = False):
        """

        :param keep_wells: Keep WELL field in output features
        """
        super().__init__()
        self._transf = {}
        self.keep_wells = keep_wells

    def fit(self, X: pd.DataFrame, y=None):

        # Categorial encoders
        for feat in self._cat_feat:
            if feat in X:
                ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
                self._transf[feat] = ohe.fit(X.loc[~X[feat].isna(), [feat]])

        # Labels and confidence
        for feat, encoder in zip([self._label_feat, self._conf_feat], [self.encode_labels, self.encode_confidence]):
            if feat in X:
                le = FunctionTransformer(func=encoder, validate=True)
                valid_mask = ~X[feat].isna()
                self._transf[feat] = le.fit(X.loc[valid_mask, [feat]])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        out = {}

        if self.keep_wells:
            out[self._well_feat] = X[self._well_feat]

        # Real-valued features pass-through
        self._transform_real_passthrough(X, out)

        # Categorial encoders
        self._transform_categorical(X, out)

        # Labels and confidence
        self._transform_labels_confidence(X, out)

        return pd.DataFrame(out, index=X.index)

    def _transform_real_passthrough(self, X, out):
        for feat in self._real_feat:
            if feat in X:
                out[feat] = X[feat].to_numpy(dtype=np.float32)

    def _transform_categorical(self, X, out):
        for feat in self._cat_feat:
            if feat in X:
                feat_group = np.zeros((len(X), len(self._transf[feat].categories_[0])), dtype=np.bool)
                valid_mask = ~X[feat].isna()
                if sum(valid_mask):
                    feat_group_val = self._transf[feat].transform(X.loc[valid_mask, [feat]])
                    feat_group[valid_mask] = feat_group_val
                for feat_idx in range(feat_group.shape[1]):
                    out[f'{feat}_ohe{feat_idx:02d}'] = feat_group[:, feat_idx]

    def _transform_labels_confidence(self, X, out):
        feat = self._label_feat
        if feat in X:
            out[feat] = self._transf[feat].transform(X[[feat]]).astype(np.int).ravel()

        feat = self._conf_feat
        if feat in X:
            valid_mask = ~X[feat].isna()
            out[feat] = np.ones(len(X), dtype=np.float32) * np.nan
            if sum(valid_mask):
                out[feat][valid_mask] = self._transf[feat].transform(X.loc[valid_mask, [feat]]).ravel()

    @staticmethod
    def encode_confidence(c: np.ndarray):
        """
        Encode confidence from the 1,2,3 system provided to a 1,0.75,0.5
        """
        c = np.array(c)
        return 1.25 - 0.25 * c

    @staticmethod
    def encode_labels(codes: np.ndarray):
        """
        Encode labels as from starter notebook, to be compatible with score function
        """
        codes = np.array(codes)
        labels = np.array([lithology.code2label[i] for i in codes.ravel()])
        labels.shape = codes.shape
        return labels
