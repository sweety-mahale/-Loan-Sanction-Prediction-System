import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
)

from src.entity.config_entity import ModelEvaluationConfig
from src.utils.common import load_object, save_json
from src.logger import logger
from src.exception import CustomException


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def initiate_model_evaluation(
        self,
        clf_test_path: Path,
        reg_test_path: Path,
        classifier_path: Path,
        regressor_path: Path,
    ):
        logger.info(">>> Model Evaluation started")
        try:
            # ── Classification evaluation ────────────────────────────────────
            clf_test = pd.read_csv(clf_test_path)
            X_test_clf = clf_test.iloc[:, :-1].values
            y_test_clf = clf_test.iloc[:, -1].values.astype(int)

            classifier = load_object(str(classifier_path))
            y_pred_clf = classifier.predict(X_test_clf)

            clf_metrics = {
                "accuracy": round(float(accuracy_score(y_test_clf, y_pred_clf)), 4),
                "precision": round(float(precision_score(y_test_clf, y_pred_clf, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test_clf, y_pred_clf, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_test_clf, y_pred_clf, zero_division=0)), 4),
            }
            logger.info(f"Classifier metrics: {clf_metrics}")

            # ── Regression evaluation ────────────────────────────────────────
            reg_test = pd.read_csv(reg_test_path)
            X_test_reg = reg_test.iloc[:, :-1].values
            y_test_reg = reg_test.iloc[:, -1].values

            regressor = load_object(str(regressor_path))
            y_pred_reg = regressor.predict(X_test_reg)

            reg_metrics = {
                "r2_score": round(float(r2_score(y_test_reg, y_pred_reg)), 4),
                "mae": round(float(mean_absolute_error(y_test_reg, y_pred_reg)), 2),
                "rmse": round(float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))), 2),
            }
            logger.info(f"Regressor metrics: {reg_metrics}")

            # ── Save combined metrics ────────────────────────────────────────
            all_metrics = {
                "classification": clf_metrics,
                "regression": reg_metrics,
            }
            save_json(path=self.config.metrics_path, data=all_metrics)

            logger.info(">>> Model Evaluation completed")
            return all_metrics

        except Exception as e:
            raise CustomException(e, sys)
