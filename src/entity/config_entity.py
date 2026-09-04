import os

from pymongo import MongoClient
from src.utils.common import Manutils
from src.constants.prediction_pipeline import PRED_SCHEMA_FILE_PATH
from src.constants import prediction_pipeline
from src.constants.training_pipeline import *
from pymongo import MongoClient
from pymongo import MongoClient
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(PIPELINE_NAME,ARTIFACT_DIR,TIMESTAMP)
    timestamp:str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    feature_store_file_path:str = os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    ingested_data_dir:str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR)
    training_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
    train_test_split_ratio: str = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataTransformationConfig:
    data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DIR, TRAIN_FILE_NAME.replace(".csv",".npz"))
    transformed_test_file_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DIR, TEST_FILE_NAME.replace(".csv",".npz"))
    transformed_object_file_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DIR, TRANSFORMED_OBJECT_FILE_NAME)

@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    valid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_VALID_DIR)
    invalid_data_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_INVALID_DIR)
    valid_train_file_path: str = os.path.join(valid_data_dir,TRAIN_FILE_NAME)
    valid_test_file_path: str = os.path.join(valid_data_dir,TEST_FILE_NAME)
    invalid_train_file_path: str = os.path.join(invalid_data_dir,TRAIN_FILE_NAME)
    invalid_test_file_path: str = os.path.join(invalid_data_dir,TEST_FILE_NAME)
    drift_report_file_path: str = os.path.join(data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

    


class PCAconfig:
    def __init__(self):
        self.n_components = 2
        self.random_state = 42

    def get_pca_config(self):
        return self.__dict__

