from pandas import DataFrame
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logging import logger
import os, sys

from dataclasses import dataclass

class CustomerSegmentationModel:
    def __init__(self,preprocessing_object: Pipeline,trained_model_object:object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self,dataframe: DataFrame)-> DataFrame:
        logger.info("Entered predict method of CustoMerSegmentation class")
        try:
            logger.info("Using the trained model to get prediction")
            transformed_feature = self.preprocessing_object.transform(dataframe)

            logger.info("used trained model to get prediction")
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CustomException(e,sys) from e

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"