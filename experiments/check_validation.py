import numpy as np

from src.entity.config_entity import (DataIngestionConfig,DataValidationConfig)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


ia = DataIngestion(DataIngestionConfig()).initial_data_ingestion()
va = DataValidation(ia, DataValidationConfig()).initiate_data_validation()
print("Validation status:", va.validation_status)
print(va)
