from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.logging import logger
import numpy as np

import sys
import pandas as pd

from src.ml.model.local_estimator import LocalModelEstimator
from dataclasses import dataclass
from typing import Optional
from src.entity.config_entity import Prediction_config

from src.utils.common import Mainutils,load_numpy_array
from src.ml.metric import calculate_metric
from src.entity.artifact_entity import ClassificationMetricArtifact

@dataclass
class EvaluationModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    changed_accuracy: float
    best_model_metric_artifact: ClassificationMetricArtifact

    @staticmethod
    def convert_test_numpy_array_to_dataframe(array:np.ndarray)-> pd.DataFrame:
        '''convert numpy array to dataframe'''
        prediction_config = Prediction_config().__dict__
        columns = prediction_config['prediction_schema']['columns'].keys()

        dataframe = pd.DataFrame(array, columns= columns)
        return dataframe


class ModelEvaluation:

    def __init__(self,model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact
                ,model_trainer_artifact:ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.utils  = Mainutils()

        except Exception as e:
            raise CustomException(e,sys) from e


    def load_trained_model(self) -> Optional[LocalModelEstimator]:

        try:
            import os
            os.path.join(self.model_eval_config.best_model_dir, self.model_eval_config.best_model_file_name)
            return None
            trained_model_object = self.utils.load_object(
                file_path=self.model_eval_config.best_model_file_name
            )

            local_model_estimator = LocalModelEstimator(model_path = trained_model_object)
            
            return local_model_estimator

        except Exception as e:
            raise CustomException(e,sys) from e

    def evaluate_model(self)-> EvaluationModelResponse:
        try:
            test_arr = load_numpy_array(
                file_path = self.data_transformation_artifact.transformed_test_file_path)

            x_test, y_test = pd.DataFrame(test_arr[:,:-1]),pd.DataFrame(test_arr[:,-1])
            x_test = EvaluationModelResponse.convert_test_numpy_array_to_dataframe(array = x_test)

            trained_model = self.utils.load_object(file_path = self.model_trainer_artifact.trained_model_file_path)

            y_hat_trained_model = trained_model.predict(x_test)

            trained_model_f1_score = f1_score(y_test,y_hat_trained_model,average= "weighted")
            best_model_f1_score = None
            best_model_metric_artifact = None
            best_model = self.load_trained_model()

            if best_model is not None:
                y_hat_trained_model = best_model.predict(x_test)
                best_model_f1_score = f1_score(y_test,y_hat_trained_model,average= "weighted")
                best_model_metric_artifact = calculate_metric(best_model,x_test,y_test)
                # Calculate how much percentage of trained model accuracy is increased / decrease
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score

            result = EvaluationModelResponse(trained_model_f1_score= trained_model_f1_score,
                                            best_model_f1_score = best_model_f1_score,
                                            is_model_accepted= trained_model_f1_score > tmp_best_model_score,
                                            changed_accuracy= trained_model_f1_score - tmp_best_model_score,
                                            best_model_metric_artifact= best_model_metric_artifact)


            logger.info(f"result {result}")
            return result

        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            evaluate_model_respose = self.evaluate_model()
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted= evaluate_model_respose.is_model_accepted,
                best_model_path= self.model_trainer_artifact.trained_model_file_path,
                trained_model_path = self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy= evaluate_model_respose.changed_accuracy,
                best_model_metric_artifact= evaluate_model_respose.best_model_metric_artifact)


            logger.info(f"ModelEvaluation Artifact {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e,sys)





                    
                                                







    
