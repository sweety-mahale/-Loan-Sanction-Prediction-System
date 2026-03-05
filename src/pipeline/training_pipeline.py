import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src.logger import logger
from src.exception import CustomException


def run_training_pipeline():
    """Orchestrates all 5 pipeline stages end-to-end."""
    try:
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("=" * 60)

        config = ConfigurationManager()

        # Stage 1: Data Ingestion
        logger.info("Stage 1/5 — Data Ingestion")
        data_ingestion = DataIngestion(config=config.get_data_ingestion_config())
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Stage 2: Data Validation
        logger.info("Stage 2/5 — Data Validation")
        data_validation = DataValidation(config=config.get_data_validation_config())
        validation_status = data_validation.initiate_data_validation(train_path, test_path)
        if not validation_status:
            logger.warning("Data validation failed. Proceeding with caution.")

        # Stage 3: Data Transformation
        logger.info("Stage 3/5 — Data Transformation")
        data_transformation = DataTransformation(config=config.get_data_transformation_config())
        (
            clf_train_path, clf_test_path,
            reg_train_path, reg_test_path,
            preprocessor_clf_path, preprocessor_reg_path,
        ) = data_transformation.initiate_data_transformation(train_path, test_path)

        # Stage 4: Model Training
        logger.info("Stage 4/5 — Model Training")
        model_trainer = ModelTrainer(config=config.get_model_trainer_config())
        classifier_path, regressor_path = model_trainer.initiate_model_training(
            clf_train_path, reg_train_path
        )

        # Stage 5: Model Evaluation
        logger.info("Stage 5/5 — Model Evaluation")
        model_evaluation = ModelEvaluation(config=config.get_model_evaluation_config())
        metrics = model_evaluation.initiate_model_evaluation(
            clf_test_path, reg_test_path, classifier_path, regressor_path
        )

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Metrics: {metrics}")
        logger.info("=" * 60)

        return metrics

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
