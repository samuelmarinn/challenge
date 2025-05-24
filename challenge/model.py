import numpy as np
import pandas as pd

from typing import Tuple, Union, List

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from challenge.pandas_utils import get_min_diff
from challenge.settings import BEST_MODEL, FTS_COLNAMES, RANDOM_STATE, TEST_SIZE, TOP_10_FTS

class DelayModel:

    def __init__(
        self, model=BEST_MODEL
    ):
        self._model = model

    def preprocess(
        self,
        data: pd.DataFrame,
        fts_cols: List[str]=FTS_COLNAMES,
        target_column: str=None,
        threshold_in_minutes: int=15,
        top_fts: List[str]=TOP_10_FTS
    ) -> Tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            fts_cols (List(str)): List of features column names
            target_column (str, optional): if set, the target is returned. Defaults to FTS_COLNAMES
            threshols_in_minutes (int): Max delay of flights. Defaults to 15
            top_10_fts (List(str)): List of top features columns. Defaults to TOP_10_FTS

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data['min_diff'] = data.apply(
            get_min_diff, args=('Fecha-O', 'Fecha-I'), axis = 1
        )
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        training_data = shuffle(
            data[fts_cols], random_state=111
        )

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )

        features_tr = features[top_fts]

        if target_column is None:
            return features_tr
        
        else:

            return [features_tr, pd.DataFrame(training_data[target_column])]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
        test_size: float=TEST_SIZE,
        random_state: int=RANDOM_STATE
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
            test_size(float): test/train size split. Defaults to constant TEST_SIZE
            random_state(int): Random seed. Defaults to RANDOM_STATE constant
        """
        x_train, _, y_train, _ = train_test_split(
            features, target[target.columns[0]], test_size = test_size, random_state = random_state
        ) ###added to avoid overfitting

        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])

        self._model.set_params(class_weight={1: n_y0/len(y_train.index), 0: n_y1/len(y_train.index)})
        self._model.fit (x_train, y_train)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        predictions = [int(x) for x in self._model.predict(features)]
        return predictions
    