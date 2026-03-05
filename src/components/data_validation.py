import sys
import pandas as pd
from pathlib import Path

from src.entity.config_entity import DataValidationConfig
from src.utils.common import read_yaml
from src.constants import SCHEMA_FILE_PATH
from src.logger import logger
from src.exception import CustomException


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema = read_yaml(SCHEMA_FILE_PATH)

    def validate_all_columns(self, df: pd.DataFrame) -> bool:
        """Check that all expected columns are present."""
        try:
            validation_status = True
            all_columns = self.schema.columns
            missing_cols = []

            for col in all_columns:
                if col not in df.columns:
                    missing_cols.append(col)
                    validation_status = False

            with open(self.config.validation_status_file, "w") as f:
                if validation_status:
                    f.write("Validation Status: True\nAll columns present.\n")
                    logger.info("Data validation passed — all columns present.")
                else:
                    f.write(
                        f"Validation Status: False\nMissing columns: {missing_cols}\n"
                    )
                    logger.warning(f"Data validation failed — missing: {missing_cols}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, train_path: Path, test_path: Path) -> bool:
        logger.info(">>> Data Validation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_status = self.validate_all_columns(train_df)
            test_status = self.validate_all_columns(test_df)

            logger.info(">>> Data Validation completed")
            return train_status and test_status

        except Exception as e:
            raise CustomException(e, sys)
