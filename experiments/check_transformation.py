import numpy as np

from src.entity.config_entity import (DataIngestionConfig, DataValidationConfig,
                                      DataTransformationConfig)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

ia = DataIngestion(DataIngestionConfig()).initial_data_ingestion()
va = DataValidation(ia, DataValidationConfig()).initiate_data_validation()
print("Validation status:", va.validation_status)

ta = DataTransformation(ia, va, DataTransformationConfig()).initial_data_transformation()
print(ta)

# Sanity checks on the saved arrays
train_arr = np.load(ta.transformed_train_file_path)
test_arr = np.load(ta.transformed_test_file_path)
print("Train shape:", train_arr.shape, "| Test shape:", test_arr.shape)
print("Train cluster counts:", np.bincount(train_arr[:, -1].astype(int)))
print("Test cluster counts: ", np.bincount(test_arr[:, -1].astype(int)))
print("NaNs in train:", np.isnan(train_arr).sum())