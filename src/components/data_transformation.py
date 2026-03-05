import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    FunctionTransformer,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from src.entity.config_entity import DataTransformationConfig
from src.utils.common import save_object, read_yaml
from src.constants import SCHEMA_FILE_PATH
from src.logger import logger
from src.exception import CustomException

# ── Helper ──────────────────────────────────────────────────────────────────
DROP_COLS = [
    "Customer ID", "Name", "Expense Type 1", "Expense Type 2",
    "Property ID", "Property Age", "Property Type",
]
INVALID_VALUE = -999

ORDINAL_FEATURES = ["Income Stability", "Location", "Has Active Credit Card", "Property Location"]
ORDINAL_CATEGORIES = [
    ["Low", "High"],
    ["Rural", "Semi-Urban", "Urban"],
    ["Unpossessed", "Inactive", "Active"],
    ["Rural", "Semi-Urban", "Urban"],
]
NOMINAL_FEATURES = ["Gender", "Profession", "Type of Employment"]
LOG_FEATURES = ["Income (USD)", "Loan Amount Request (USD)",
                "Current Loan Expenses (USD)", "Property Price"]
NUMERIC_FEATURES = ["Age", "Credit Score", "No. of Defaults", "Dependents"]

TARGET_CLF = "Loan sanctioned"
TARGET_REG = "Loan Sanction Amount (USD)"

PROFESSION_THRESHOLD = 100
IQR_FACTOR = 1.5
IQR_COLS = [
    "Age", "Income (USD)", "Loan Amount Request (USD)",
    "Current Loan Expenses (USD)", "Credit Score",
    "Property Price", "Dependents",
]


def _log1p_transform(X):
    """Safe log(1+x) transformation."""
    return np.log1p(np.abs(X))


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # ── Private helpers ──────────────────────────────────────────────────────

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unneeded columns, remove -999 rows, fix -1 placeholders."""
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
        df = df.dropna(subset=[TARGET_REG])

        # Remove rows with -999
        mask_cols = [c for c in [
            "Current Loan Expenses (USD)", "Co-Applicant",
            "Property Price", TARGET_REG,
        ] if c in df.columns]
        if mask_cols:
            keep = (df[mask_cols] != INVALID_VALUE).all(axis=1)
            df = df.loc[keep]

        # Replace -1 placeholders
        for col in ["Income (USD)", "Current Loan Expenses (USD)", "Dependents"]:
            if col in df.columns:
                df[col] = df[col].replace({-1: 0})

        return df.reset_index(drop=True)

    def _feature_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group rare Profession categories; derive Loan sanctioned flag."""
        if "Profession" in df.columns:
            counts = df["Profession"].value_counts()
            rare = counts[counts < PROFESSION_THRESHOLD].index
            df["Profession"] = df["Profession"].replace(rare, "others")

        if TARGET_REG in df.columns:
            df[TARGET_CLF] = df[TARGET_REG] > 0

        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Column-level imputation matching notebook logic."""
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
        if "Has Active Credit Card" in df.columns:
            df["Has Active Credit Card"] = df["Has Active Credit Card"].fillna(
                df["Has Active Credit Card"].mode()[0]
            )
        if "Credit Score" in df.columns:
            df["Credit Score"] = df["Credit Score"].fillna(df["Credit Score"].median())
        if "Type of Employment" in df.columns:
            if "Profession" in df.columns:
                df.loc[df["Profession"] == "Pensioner", "Type of Employment"] = "Retired"
                df.loc[df["Profession"] == "Unemployed", "Type of Employment"] = "Unknown"
            df["Type of Employment"] = df["Type of Employment"].fillna("Not available")
        if "Property Location" in df.columns:
            df["Property Location"] = df["Property Location"].fillna(
                df["Property Location"].mode()[0]
            )
        if "Income Stability" in df.columns:
            df["Income Stability"] = df["Income Stability"].fillna(
                df["Income Stability"].mode()[0]
            )
        df = df.dropna()
        return df

    def _cap_outliers(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """IQR-based capping for the given columns."""
        for col in cols:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - IQR_FACTOR * iqr
            upper = q3 + IQR_FACTOR * iqr
            df[col] = np.clip(df[col], lower, upper)
        return df

    def _build_preprocessor(self):
        """Build sklearn ColumnTransformer pipeline."""
        log_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(_log1p_transform, validate=False)),
            ("scaler", StandardScaler()),
        ])
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=ORDINAL_CATEGORIES,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        nominal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="ignore"
            )),
        ])
        preprocessor = ColumnTransformer([
            ("log_features", log_pipeline, LOG_FEATURES),
            ("num_features", num_pipeline, NUMERIC_FEATURES),
            ("ordinal_features", ordinal_pipeline, ORDINAL_FEATURES),
            ("nominal_features", nominal_pipeline, NOMINAL_FEATURES),
        ], remainder="drop")
        return preprocessor

    # ── Public interface ─────────────────────────────────────────────────────

    def initiate_data_transformation(self, train_path: Path, test_path: Path):
        logger.info(">>> Data Transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # --- Cleaning & Engineering ---
            train_df = self._clean_dataframe(train_df)
            test_df = self._clean_dataframe(test_df)

            train_df = self._feature_engineer(train_df)
            test_df = self._feature_engineer(test_df)

            train_df = self._impute_missing(train_df)
            test_df = self._impute_missing(test_df)

            # --- Outlier capping ---
            train_df = self._cap_outliers(train_df, IQR_COLS)
            test_df = self._cap_outliers(test_df, IQR_COLS)

            # ── CLASSIFICATION BRANCH ────────────────────────────────────────
            false_count = train_df[TARGET_CLF].value_counts()[False]
            df_true = train_df[train_df[TARGET_CLF] == True].sample(
                n=min(false_count, (train_df[TARGET_CLF] == True).sum()),
                random_state=42,
            )
            df_false = train_df[train_df[TARGET_CLF] == False]
            train_clf = pd.concat([df_true, df_false]).reset_index(drop=True)

            X_train_clf = train_clf.drop(columns=[TARGET_CLF, TARGET_REG], errors="ignore")
            y_train_clf = train_clf[TARGET_CLF].astype(int)

            X_test_clf = test_df.drop(columns=[TARGET_CLF, TARGET_REG], errors="ignore")
            y_test_clf = test_df[TARGET_CLF].astype(int)

            preprocessor_clf = self._build_preprocessor()
            X_train_clf_arr = preprocessor_clf.fit_transform(X_train_clf)
            X_test_clf_arr = preprocessor_clf.transform(X_test_clf)

            clf_train = np.c_[X_train_clf_arr, y_train_clf.values]
            clf_test = np.c_[X_test_clf_arr, y_test_clf.values]

            pd.DataFrame(clf_train).to_csv(self.config.clf_train_path, index=False)
            pd.DataFrame(clf_test).to_csv(self.config.clf_test_path, index=False)
            save_object(str(self.config.preprocessor_clf_path), preprocessor_clf)
            logger.info("Classification preprocessor saved.")

            # ── REGRESSION BRANCH ────────────────────────────────────────────
            train_reg = train_df[train_df[TARGET_REG] > 0].drop(
                columns=[TARGET_CLF], errors="ignore"
            )
            test_reg = test_df[test_df[TARGET_REG] > 0].drop(
                columns=[TARGET_CLF], errors="ignore"
            )

            # Additional outlier capping for regression target
            train_reg = self._cap_outliers(train_reg, IQR_COLS + [TARGET_REG])
            test_reg = self._cap_outliers(test_reg, IQR_COLS + [TARGET_REG])

            X_train_reg = train_reg.drop(columns=[TARGET_REG], errors="ignore")
            y_train_reg = train_reg[TARGET_REG]

            X_test_reg = test_reg.drop(columns=[TARGET_REG], errors="ignore")
            y_test_reg = test_reg[TARGET_REG]

            preprocessor_reg = self._build_preprocessor()
            X_train_reg_arr = preprocessor_reg.fit_transform(X_train_reg)
            X_test_reg_arr = preprocessor_reg.transform(X_test_reg)

            reg_train = np.c_[X_train_reg_arr, y_train_reg.values]
            reg_test = np.c_[X_test_reg_arr, y_test_reg.values]

            pd.DataFrame(reg_train).to_csv(self.config.reg_train_path, index=False)
            pd.DataFrame(reg_test).to_csv(self.config.reg_test_path, index=False)
            save_object(str(self.config.preprocessor_reg_path), preprocessor_reg)
            logger.info("Regression preprocessor saved.")

            logger.info(">>> Data Transformation completed")
            return (
                self.config.clf_train_path, self.config.clf_test_path,
                self.config.reg_train_path, self.config.reg_test_path,
                self.config.preprocessor_clf_path, self.config.preprocessor_reg_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
