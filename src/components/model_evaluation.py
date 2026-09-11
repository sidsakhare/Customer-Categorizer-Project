from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.pipeline.prediction_pipeline import CustomException
from src.constants.training_pipeline import TARGET_COLUMN
from src.logging import logger

import sys
import pandas as pd

from src.ml.model.estimator import CustomerSegmentationModel
from dataclasses import dataclass
from typing import Optional
from src.entity.config_entity import Prediction_config

from src.utils.main_utils import MainUtils,load_numpy_array_data
from src.ml.metric import calculate_metric
from src.entity.artifact_entity import ClassificationMetricArtifact

@dataclass
class EvaluationModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    changed_accuracy: float
    best_model_metric_artifact: ClassificationMetricArtifact


    def convert_test_numpy_array_to_dataframe(array:str):
        '''convert numpy array to dataframe'''
        prediction_config = Prediction_config().__dict__
        columns = prediction_config['prediction_schema']['columns'].keys()

        dataframe = pd.DataFrame(array, columns= columns)
        return dataframe
