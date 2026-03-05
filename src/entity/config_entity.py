from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    validation_status_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    preprocessor_clf_path: Path
    preprocessor_reg_path: Path
    clf_train_path: Path
    clf_test_path: Path
    reg_train_path: Path
    reg_test_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    classifier_path: Path
    regressor_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_path: Path
