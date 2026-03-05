import sys
from pathlib import Path

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)
from src.utils.common import read_yaml, create_directories
from src.logger import logger
from src.exception import CustomException


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        try:
            self.config = read_yaml(config_filepath)
            self.params = read_yaml(params_filepath)
            self.schema = read_yaml(schema_filepath)
            create_directories([self.config.artifacts_root])
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_path=Path(config.source_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            test_size=config.test_size,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            validation_status_file=Path(config.validation_status_file),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            preprocessor_clf_path=Path(config.preprocessor_clf_path),
            preprocessor_reg_path=Path(config.preprocessor_reg_path),
            clf_train_path=Path(config.clf_train_path),
            clf_test_path=Path(config.clf_test_path),
            reg_train_path=Path(config.reg_train_path),
            reg_test_path=Path(config.reg_test_path),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            classifier_path=Path(config.classifier_path),
            regressor_path=Path(config.regressor_path),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            metrics_path=Path(config.metrics_path),
        )
