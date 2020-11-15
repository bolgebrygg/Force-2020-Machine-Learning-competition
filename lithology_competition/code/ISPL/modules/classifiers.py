import logging
from copy import deepcopy
from typing import Set

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm

import preprocessing


class FeatDepXGBoost(BaseEstimator):

    def __init__(self, *,
                 feat_comb_set: Set[frozenset] = None,
                 feat_criteria: str = 'any',
                 pipeline: Pipeline = make_pipeline(
                     preprocessing.SimpleImputeTransformer(keep_wells=True, missing_indicator=True),
                     preprocessing.LogTransformer(),
                     preprocessing.LpfTransformer(),
                     preprocessing.DepthTransformer(),
                 ),
                 finetune_rounds: int = 40,
                 finetune_th: float = 0.7,
                 num_boost_rounds: int = 250
                 ):
        """

        :param feat_comb_set: set of features combinations
        :param feat_criteria:
            'all' feature is selected if all samples are not NaN
            'any' feature is selected if any sample is not NaN
        :param pipeline: Pipeline for preprocessing
        :param finetune_rounds: number of rounds of finetuning in test (if None no finetuning)
        :param finetune_th: confidence threshold for finetuning in test
        :param num_boost_rounds: number of boost rounds for first stage classifier
        """

        if feat_criteria not in ['all', 'any']:
            raise ValueError(f'Unknown feature criteria: {feat_criteria}')

        self._xgb_param = {
            'objective': 'multi:softprob',
            'num_class': 12,
            'tree_method': 'gpu_hist',
            'random_state': 42,
            'subsample': 0.1,
            'max_depth': 3,
            'max_delta_step': 1,
            'eta': 0.3,
            'booster': 'gbtree',
            'gamma': 1,
            'min_child_weight': 1,
            'sampling_method': 'gradient_based',
            'lambda': 1,
            'alpha': 1,
        }

        self._xgb_param_ft = {
            'objective': 'multi:softprob',
            'num_class': 12,
            'tree_method': 'gpu_hist',
            'random_state': 42,
            'subsample': 0.1,
            'max_depth': 3,
            'max_delta_step': 1,
            'eta': 0.01,
            'booster': 'gbtree',
            'gamma': 1,
            'min_child_weight': 1,
            'sampling_method': 'gradient_based',
            'lambda': 1,
            'alpha': 1,
        }

        self.feat_criteria = feat_criteria
        self.finetune_rounds = finetune_rounds
        self.finetune_th = finetune_th
        self.num_boost_rounds = num_boost_rounds
        self.feat_comb_set = {} if feat_comb_set is None else feat_comb_set
        self.pipeline = pipeline

        # To fit
        self._pipeline_map = {}
        self._pipeline_all = None

        self._xgb_map = {}
        self._xgb_all = None

    def fit(self, X, y):
        assert (isinstance(X, pd.DataFrame))

        # Fit pipeline and classifier for each combination of features
        print('Training with features subsets')
        for feat_comb in tqdm(self.feat_comb_set):
            X_comb = X.loc[:, list(feat_comb) + ['WELL']]
            self._pipeline_map[feat_comb] = deepcopy(self.pipeline)
            X_aug = self._pipeline_map[feat_comb].fit_transform(X_comb)
            self._xgb_map[feat_comb] = xgb.XGBClassifier(
                **self._xgb_param,
                n_estimators=self.num_boost_rounds
            ).fit(X_aug.drop(['WELL'], axis=1), y)

        print('Training with all features')
        # Fit pipeline and classifier for all features (fallback solution)
        self._pipeline_all = deepcopy(self.pipeline)
        X_aug = self._pipeline_all.fit_transform(X)
        self._xgb_all = xgb.XGBClassifier(
            **self._xgb_param,
            n_estimators=self.num_boost_rounds
        ).fit(X_aug.drop(['WELL'], axis=1), y)

        return self

    def get_clf_feat_per_well(self, X_well: pd.DataFrame) -> dict:
        assert (isinstance(X_well, pd.DataFrame))
        assert (X_well['WELL'].nunique() == 1)

        if self.feat_criteria == 'all':
            feat_comb_mask = X_well.notna().all(axis=0)
        elif self.feat_criteria == 'any':
            feat_comb_mask = X_well.notna().any(axis=0)
        else:
            raise RuntimeError()

        feat_comb = frozenset(feat_comb_mask[feat_comb_mask].index.drop('WELL'))

        if feat_comb in self.feat_comb_set:
            logging.debug('XGBoost trained on: ' + ', '.join(feat_comb))
            X_comb = X_well.loc[:, list(feat_comb) + ['WELL']]
            X_aug = self._pipeline_map[feat_comb].transform(X_comb)
            clf = self._xgb_map[feat_comb]
        else:
            logging.debug('XGBoost trained on all features')
            X_aug = self._pipeline_all.transform(X_well)
            clf = self._xgb_all

        feat = X_aug.drop(['WELL'], axis=1)

        return {
            'clf': clf,
            'feat': feat
        }

    def predict_proba(self, X) -> np.ndarray:
        assert (isinstance(X, pd.DataFrame))

        proba = pd.DataFrame(np.zeros((len(X), 12)), index=X.index)
        for well in X['WELL'].unique():
            X_well = X.loc[X['WELL'] == well]
            combo = self.get_clf_feat_per_well(X_well)
            well_proba = combo['clf'].predict_proba(combo['feat'])

            if self.finetune_rounds is not None:
                well_pred = np.argmax(well_proba, axis=1)
                well_conf = np.max(well_proba, axis=1)

                clf_ft = xgb.XGBClassifier(
                    **self._xgb_param_ft,
                    n_estimators=self.finetune_rounds
                ).fit(
                    combo['feat'][well_conf > self.finetune_th],
                    well_pred[well_conf > self.finetune_th],
                    xgb_model=combo['clf'].get_booster()
                )

                well_proba = clf_ft.predict_proba(combo['feat'])

            proba.loc[X_well.index] = well_proba

        return proba.to_numpy()

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
