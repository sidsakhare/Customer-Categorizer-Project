from src.components.model_trainer import ModelTrainer
from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                      DataTransformationConfig, ModelTrainerConfig)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


ia = DataIngestion(DataIngestionConfig()).initial_data_ingestion()
va = DataValidation(ia, DataValidationConfig()).initiate_data_validation()
ta = DataTransformation(ia, va, DataTransformationConfig()).initial_data_transformation()
print("Transformation done:", ta)

mt = ModelTrainer(data_transformation_artifact=ta,
                  model_trainer_config=ModelTrainerConfig()).initiate_model_trainer()
print(mt)