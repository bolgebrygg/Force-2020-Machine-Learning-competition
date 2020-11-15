import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from . import BaseTransformer


class SimpleImputeTransformer(BaseTransformer):
    """
    - One-hot encoding of categorial features (FORMATION,GROUP). Null samples are encoded as all-zeros.
    - Labels encoding according to the challenge score function
    - Confidence encoding from the 1,2,3 system of the challenge to 1,0.75,0.5
    - Missing values of real-valued features are imputed with the selected method.
      An additional missing indicator is added.
    """

    def __init__(self, *, strategy: str = 'median', missing_indicator: bool = False, keep_wells: bool = False):
        """

        :param strategy: SimpleImputer strategy
        :param missing_indicator: Add missing indicator to real-valued features
        :param keep_wells: Keep WELL field in output features
        """
        super().__init__(keep_wells=keep_wells)
        self.strategy = str(strategy)
        self.missing_indicator = bool(missing_indicator)

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X)

        # Real-valued features imputation
        feat_list = [f for f in self._real_feat if f in X]
        self._transf['real'] = SimpleImputer(strategy=self.strategy).fit(X[feat_list].astype(np.float32))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        out = {}

        if self.keep_wells:
            out[self._well_feat] = X[self._well_feat]

        # Real-valued features pass-through
        self._transform_real_impute(X, out)

        # Categorial encoders
        self._transform_categorical(X, out)

        # Labels and confidence
        self._transform_labels_confidence(X, out)

        return pd.DataFrame(out, index=X.index)

    def _transform_real_impute(self, X, out):
        feat_list = [f for f in self._real_feat if f in X]
        imputed = self._transf['real'].transform(X[feat_list]).astype(np.float32)
        for feat_idx, feat in enumerate(feat_list):
            out[feat] = imputed[:, feat_idx]
            if self.missing_indicator:
                out[f'{feat}_miss'] = X[feat].isna()
