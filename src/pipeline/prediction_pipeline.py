import sys
import pandas as pd
import numpy as np

from src.utils.common import load_object
from src.logger import logger
from src.exception import CustomException

# Paths to saved artifacts
PREPROCESSOR_CLF_PATH = "artifacts/data_transformation/preprocessor_clf.pkl"
PREPROCESSOR_REG_PATH = "artifacts/data_transformation/preprocessor_reg.pkl"
CLASSIFIER_PATH = "artifacts/model_trainer/classifier.pkl"
REGRESSOR_PATH = "artifacts/model_trainer/regressor.pkl"

# ── Columns dropped during ingestion ────────────────────────────────────────
DROP_COLS_ALWAYS = [
    "Customer ID", "Name", "Expense Type 1", "Expense Type 2",
    "Property ID", "Property Age", "Property Type",
    "Loan sanctioned", "Loan Sanction Amount (USD)",
]


class PredictPipeline:
    def __init__(self):
        try:
            self.preprocessor_clf = load_object(PREPROCESSOR_CLF_PATH)
            self.preprocessor_reg = load_object(PREPROCESSOR_REG_PATH)
            self.classifier = load_object(CLASSIFIER_PATH)
            self.regressor = load_object(REGRESSOR_PATH)
            logger.info("Models loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Predict loan sanction and amount.
        Returns {"sanctioned": bool, "amount": float or None}
        """
        try:
            # Drop any unexpected columns
            features = features.drop(
                columns=[c for c in DROP_COLS_ALWAYS if c in features.columns],
                errors="ignore",
            )

            # Fix -1 placeholders
            for col in ["Income (USD)", "Current Loan Expenses (USD)", "Dependents"]:
                if col in features.columns:
                    features[col] = features[col].replace({-1: 0})

            # Classification
            X_clf = self.preprocessor_clf.transform(features)
            loan_sanctioned = bool(self.classifier.predict(X_clf)[0])
            logger.info(f"Classification result: {loan_sanctioned}")

            # Regression (only if sanctioned)
            amount = None
            if loan_sanctioned:
                X_reg = self.preprocessor_reg.transform(features)
                amount = float(self.regressor.predict(X_reg)[0])
                amount = max(0.0, round(amount, 2))
                logger.info(f"Regression result: ${amount:,.2f}")

            return {"sanctioned": loan_sanctioned, "amount": amount}

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Maps web form inputs to a feature DataFrame."""

    def __init__(
        self,
        age: float,
        gender: str,
        income: float,
        income_stability: str,
        profession: str,
        employment_type: str,
        location: str,
        loan_amount_request: float,
        current_loan_expenses: float,
        credit_card_status: str,
        property_location: str,
        credit_score: float,
        no_of_defaults: int,
        dependents: int,
        property_price: float,
    ):
        self.age = age
        self.gender = gender
        self.income = income
        self.income_stability = income_stability
        self.profession = profession
        self.employment_type = employment_type
        self.location = location
        self.loan_amount_request = loan_amount_request
        self.current_loan_expenses = current_loan_expenses
        self.credit_card_status = credit_card_status
        self.property_location = property_location
        self.credit_score = credit_score
        self.no_of_defaults = no_of_defaults
        self.dependents = dependents
        self.property_price = property_price

    def get_data_as_dataframe(self) -> pd.DataFrame:
        data = {
            "Age": [self.age],
            "Gender": [self.gender],
            "Income (USD)": [self.income],
            "Income Stability": [self.income_stability],
            "Profession": [self.profession],
            "Type of Employment": [self.employment_type],
            "Location": [self.location],
            "Loan Amount Request (USD)": [self.loan_amount_request],
            "Current Loan Expenses (USD)": [self.current_loan_expenses],
            "Has Active Credit Card": [self.credit_card_status],
            "Property Location": [self.property_location],
            "Credit Score": [self.credit_score],
            "No. of Defaults": [self.no_of_defaults],
            "Dependents": [self.dependents],
            "Property Price": [self.property_price],
        }
        return pd.DataFrame(data)
