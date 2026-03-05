import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.entity.config_entity import ModelTrainerConfig
from src.utils.common import save_object, read_yaml
from src.constants import PARAMS_FILE_PATH
from src.logger import logger
from src.exception import CustomException


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.params = read_yaml(PARAMS_FILE_PATH)

    def initiate_model_training(
        self,
        clf_train_path: Path,
        reg_train_path: Path,
    ):
        logger.info(">>> Model Training started")
        try:
            # ── Classification ───────────────────────────────────────────────
            clf_train = pd.read_csv(clf_train_path)
            X_train_clf = clf_train.iloc[:, :-1].values
            y_train_clf = clf_train.iloc[:, -1].values.astype(int)

            clf_params = self.params.RandomForestClassifier
            classifier = RandomForestClassifier(
                n_estimators=clf_params.n_estimators,
                random_state=clf_params.random_state,
                class_weight=clf_params.class_weight,
            )
            classifier.fit(X_train_clf, y_train_clf)
            save_object(str(self.config.classifier_path), classifier)
            logger.info(f"Classifier trained & saved to {self.config.classifier_path}")

            # ── Regression ───────────────────────────────────────────────────
            reg_train = pd.read_csv(reg_train_path)
            X_train_reg = reg_train.iloc[:, :-1].values
            y_train_reg = reg_train.iloc[:, -1].values

            reg_params = self.params.RandomForestRegressor
            regressor = RandomForestRegressor(
                n_estimators=reg_params.n_estimators,
                random_state=reg_params.random_state,
            )
            regressor.fit(X_train_reg, y_train_reg)
            save_object(str(self.config.regressor_path), regressor)
            logger.info(f"Regressor trained & saved to {self.config.regressor_path}")

            logger.info(">>> Model Training completed")
            return self.config.classifier_path, self.config.regressor_path

        except Exception as e:
            raise CustomException(e, sys)
