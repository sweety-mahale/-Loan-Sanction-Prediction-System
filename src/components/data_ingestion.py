import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.logger import logger
from src.exception import CustomException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """Reads raw CSV, saves train/test splits."""
        logger.info(">>> Data Ingestion started")
        try:
            df = pd.read_csv(self.config.source_path)
            logger.info(f"Dataset loaded: shape={df.shape}")

            os.makedirs(self.config.root_dir, exist_ok=True)

            train_set, test_set = train_test_split(
                df, test_size=self.config.test_size, random_state=42
            )

            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)

            logger.info(
                f"Train size: {train_set.shape}, Test size: {test_set.shape}"
            )
            logger.info(">>> Data Ingestion completed")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
