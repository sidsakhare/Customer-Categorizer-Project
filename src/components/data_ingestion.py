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
from src.utils.common import MainUtils

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig = DataIngestionConfig()):
        self.data_ingestion_config = data_ingestion_config
        self.utils = MainUtils()

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
            train_set, test_set = train_test_split(dataframe, test_data = self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on a dataframe")

            ingested_data_dir = sself.data_ingestion_config.data_ingestion_dir
            os.makedirs(ingested_data_dir,exist_ok= True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False, header = True)
            logger.info("training data has been saved")
            test_set.to_csv(self.data_ingestion_dir.testing_file_path, index = False, headers = True)
            logger.info('Test has been saved')

        except Exception as e:
            raise CustomerData(e,sys) from e


