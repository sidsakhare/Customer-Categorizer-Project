from src.ml.model.estimator import CustomerSegmentationModel
from src.logging import logger
from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from src.constants.training_pipeline import *
from src.entity.config_entity import training_pipeline_config
from src.entity.config_entity import Prediction_config, PredictionPipelineConfig

from src.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from src.utils.common import Mainutils

from src.exception import CustomException
import pandas as pd
import numpy as np
import sys
from pandas import DataFrame

class CustomerData:
    def __init__(self):
        pass

    def get_input_dataset(self, column_schema:dict, input_data: list):
        columns = list(column_schema.keys())
        input_data = pd.DataFrame([input_data],columns= columns)

        for key, value in column_schema.items():
            input_data[key] = input_data[key].astype(value)

        return input_data

    @staticmethod

    def from_input_dataframe(data):
        Prediction_configs = Prediction_config()
        prediction_schema = Prediction_configs.__dict__
        column_schema = prediction_schema["prediction_schema"]["columns"]

        CustomerDatas = CustomerData()
        input_dataset = CustomerDatas.get_input_dataset(
            column_schema= column_schema,
            input_data= data
        )
        return input_dataset

class PredictionPipeline:
    def __init__(self):
        self.utils = Mainutils()

    def prepare_input_data(self,input_data:list) -> DataFrame:
        logger.info("Entered prepare_input_data method")
        """ 
        method: prepare_input_data 
        
        objective: This method creates a dataframe taking the column names from prediction schema file
                       with the input values for prediction and returns it

        Args:
            input_data (list): input data 

        Raises:
            Customexception

        Returns:
            customerDataframe: pd.DataFrame: a dataframe containing the input values
        """

        try:
            customerdataframe = CustomerData.from_input_dataframe(data = input_data)
            return customerdataframe

        except Exception as e:
            raise CustomException(e,sys) from e




    def get_trained_model(self) ->CustomerSegmentationModel :
        """
        method: get_trained_model

        objective: loads the preprocessing object and the trained model object
                   separately, and wraps them into a CustomerSegmentationModel

        Returns:
            CustomerSegmentationModel: object with .predict() that transforms
            the dataframe then runs inference
        """
        try:
            transformation_config  = DataTransformationConfig()
            training_config = ModelTrainerConfig()


            preprocessing_object = self.utils.load_object(
                file_path=transformation_config .transformed_object_file_path
            )
            trained_model_object = self.utils.load_object(
                file_path=training_config.trained_model_file_path
            )

            model = CustomerSegmentationModel(
                preprocessing_object=preprocessing_object,
                trained_model_object=trained_model_object
            )
            return model

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self,input_data:list):
        '''
         method: run_pipeline
        
        objective: run_pipeline method runs the whole prediction pipeline.

        Raises:
            CustomerException: 
        '''
        try:
            input_dataframe = self.prepare_input_data(input_data)
            model = self.get_trained_model()
            prediction = model.predict(input_dataframe)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
