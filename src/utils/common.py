import os
import sys
import json
import dill
import yaml
import numpy as np
from box import ConfigBox
from pathlib import Path
from ensure import ensure_annotations
from typing import Any

from src.logger import logger
from src.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads YAML file and returns a ConfigBox object."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates a list of directories."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_object(file_path: str, obj: Any):
    """Saves a Python object using dill."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """Loads a Python object using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def save_json(path: Path, data: dict):
    """Saves a dict as JSON."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_json(path: Path) -> ConfigBox:
    """Loads JSON and returns a ConfigBox."""
    try:
        with open(path) as f:
            content = json.load(f)
        return ConfigBox(content)
    except Exception as e:
        raise CustomException(e, sys)
