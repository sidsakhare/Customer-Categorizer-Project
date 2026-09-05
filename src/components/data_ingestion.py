import sys
from typing import Tuple 
import os
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.constants import DATABASE_NAME, COLLECTION_NAME
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.customer_data import CustomerData
from src.exception import CustomException
from src.logging import logger
from src.utils.common import Mainutils

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = Mainutils()





    def split_data_as_train_test(self,dataframe: DataFrame) -> Tuple[DataFrame, DataFrame]:
        '''
         Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created 
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   1.0

        '''
        logger.info('Entered split_data-as_tarin_test method od DataIngestion class')
        try:
            train_set, test_set = train_test_split(dataframe, test_size = self.data_ingestion_config.train_test_split_ratio)
            logger.info("Performed train test split on a dataframe")

            ingested_data_dir = self.data_ingestion_config.ingested_data_dir
            os.makedirs(ingested_data_dir,exist_ok= True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            logger.info("training data has been saved")
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index = False, header = True)
            logger.info('Test has been saved')

        except Exception as e:
            raise CustomException(e,sys) from e


    def export_data_into_feature_store(self)-> DataFrame:
        '''
        Method name: export_data_info_feature_store
        Description: This method exports the entire data from mongo db to pandas dataframe and saves it in feature store

        output: pandas dataframe
        on failure: raise exception and log the error

        version: 1.0
        '''
        try:
            logger.info(f"Exporting data from mongo db to pandas dataframe and saving it in feature store")
            customer_data = CustomerData()
            customer_dataframe = customer_data.export_collection_as_dataframe(collection_name = COLLECTION_NAME , database_name = DATABASE_NAME)

            logger.info(f"shape of dataframe {customer_dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok = True)
            logger.info(f"saveing dataframe to feature store dir:{feature_store_file_path}")
            customer_dataframe.to_csv(feature_store_file_path, index = False, header = True)
            return customer_dataframe
        except Exception as e:
            raise CustomException(e,sys) from e

    

    def initial_data_ingestion(self)-> DataIngestionArtifact:
        '''
        Method name: initial_data_ingestion
        Description : This method initiates data ingestion component of training pipeline 

        output: train set and test set are returned as the artifact of data ingestion component
        on failure: raise exception and log the error

        version: 1.0
        '''

        logger.info(f"Entered initial_data_ingestion method of DataIngestion class")


        try:
            dataframe = self.export_data_into_feature_store()
            schema_config = self.utils.read_schema_config_file()
            dataframe = dataframe.drop(columns = schema_config['drop_columns'])

            logger.info(f"Got the data from mongodb and saved it in feature store and dropped the columns which are not required for training")
            self.split_data_as_train_test(dataframe = dataframe)

            logger.info(f"Train test split has been performed")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path = self.data_ingestion_config.training_file_path,
                test_file_path = self.data_ingestion_config.testing_file_path
                )

            logger.info(f"Data ingestion artifact has been created: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e,sys) from e

if __name__ == "__main__":
    ingestion = DataIngestion()          # no config passed, since it builds its own
    data_ingestion_artifact = ingestion.initial_data_ingestion()
    print(f"Train saved at: {data_ingestion_artifact.trained_file_path}")
    print(f"Test saved at: {data_ingestion_artifact.test_file_path}")




