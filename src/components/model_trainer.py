import sys
from typing import List, Tuple
import os
from pandas import DataFrame
import numpy as np
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact,ClassificationMetricArtifact
from src.ml.model.estimator import CustomerSegmentationModel

from src.exception import CustomException
from src.logging import logger
from src.utils.common import Mainutils, load_numpy_array
from neuro_mf import ModelFactory
from sklearn.metrics import f1_score, recall_score, precision_score


# class CustomerSegmentationModel:
#     def __init__(self,preprocessing_object: object,trained_model_object:object):
#         self.preprocessing_object = preprocessing_object
#         self.trained_model_object = trained_model_object

#     def predict(self,dataframe: DataFrame)-> DataFrame:
#         logger.info("Entered predict method of CustoMerSegmentation class")
#         try:
#             logger.info("Using the trained model to get prediction")
#             transformed_feature = self.preprocessing_object.transform(dataframe)

#             logger.info("used trained model to get prediction")
#             return self.trained_model_object.predict(transformed_feature)

#         except Exception as e:
#             raise CustomException(e,sys) from e

#     def __repr__(self):
#         return f"{type(self.trained_model_object).__name__}()"

#     def __str__(self):
#         return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:
    def __init__(self,data_transformation_artifact: DataTransformationArtifact,model_trainer_config : ModelTrainerConfig):

        self.data_transformation_artifat = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = Mainutils()

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array(file_path = self.data_transformation_artifat.transformed_train_file_path)
            test_arr = load_numpy_array(file_path= self.data_transformation_artifat.transformed_test_file_path)
            x_train,y_train,x_test,y_test = train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]

            model_factory = ModelFactory(model_config_path= self.model_trainer_config.model_config_file_path)
            best_model_detail = model_factory.get_best_model(X = x_train,y = y_train, base_accuracy = self.model_trainer_config.expected_accuracy)
            preprocessing_obj = self.utils.load_object(file_path = self.data_transformation_artifat.transformed_object_file_path)

            if best_model_detail.best_score < self.model_trainer_config.expected_accuracy:
                logger.info("No model found with accuracy more than base model score")
                raise Esception("No best model found with score more than base score")

            customer_segmentation_model = CustomerSegmentationModel(
                preprocessing_object= preprocessing_obj,
                trained_model_object= best_model_detail.best_model
            )
            logger.info("Customer Segmentation Model is created and saved.")
            trained_model_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(trained_model_path,exist_ok= True)

            y_pred = best_model_detail.best_model.predict(x_test)
            y_pred = np.asarray(y_pred).ravel().astype(int)

            self.utils.save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= customer_segmentation_model
            )

            logger.info(f"Customer segmentation model have saved succsessfully at {trained_model_path}")
            metric_artifact = ClassificationMetricArtifact(
                f1_score= f1_score(y_test, y_pred, average="weighted"),
                precision_score= precision_score(y_test, y_pred, average="weighted"),
                recall_score= recall_score(y_test, y_pred, average="weighted")
            )
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path= self.model_trainer_config.trained_model_file_path,
                metric_artifact= metric_artifact
            )

            logger.info("Model training completed susscesfully")
            logger.info(f"Model trainer artifact {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e,sys) from e

