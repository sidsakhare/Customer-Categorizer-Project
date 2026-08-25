print("script starts here")
import shutil
import sys
from typing import Dict, Tuple
import os
import numpy as np
import pandas as pd
import pickle as pkl
from catboost import CatBoostClassifier
from sklearn.utils import all_estimators
import yaml
import importlib
from src.constants.training_pipeline import *

from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from yaml import safe_dump

from src.logging import logger

def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Load a numpy array from a file.

    Args:
        file_path (str): Path to the file containing the numpy array.
        return : np.ndarray data loaded
    """
    try:
        with open(file_path,'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomeException(e,sys) from e

def write_yaml_file(file_path:str,content:object,replace:bool = False)-> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok= True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise CustomeException(e,sys)   

class Mainutils:
    def __init__(self) ->None:
        pass

    def read_yaml_file(self,filename:str)->dict:
        try:
            with open(filename,"rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomeException(e,sys) from e

    def read_schema_config_file(self) -> dict:
        try:
            schema_config = self.read_yaml_file(SCHEMA_FILE_PATH)

            return schema_config

        except Exception as e:
            raise CustomeException(e,sys) from e

    def read_model_config_file(self) -> dict:
        try:
            model_config = self.read_yaml_file(MODEL_TRAINER_MODEL_CONFIG_FILE_PATH)

            return model_config
        except Exception as e:
            raise CustomException(e,sys) from e

    def get_tuned_model(
        self,
        model_name: str,
        train_x: DataFrame,
        train_y: DataFrame,
        test_x: DataFrame,
        test_y: DataFrame,
    ) -> Tuple[float, object, str]:

        logger.info("Entered the get_tuned_model method of MainUtils class")

        try:
            model = self.get_base_model(model_name)

            model_best_params = self.get_model_params(model, train_x, train_y)

            model.set_params(**model_best_params)

            model.fit(train_x, train_y)

            preds = model.predict(test_x)

            model_score = self.get_model_score(test_y, preds)

            logger.info("Entered the get_tuned_model method of MainUtils class")

            return model_score, model, model.__class__.__name__

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def get_base_model(model_name:str,model_config:dict)->object:
        '''
        This method is used to import model fron reading yaml file and return model
        object with base parameters 
        '''
        
        logger.info("Entered the get_base_model method of Mainutils class")

        try:
            model_config_block = model_config["model_selection"]["module_0"]
            module_path  = model_config_block['module']
            class_name = model_config_block["class"]
            base_params = model_config_block.get("params",{})

            module = importlib.import_module(module_path)
            model_class = getattr(module,class_name)
            model = model_class(**base_params)

            logger.info("Exited the get_base_model method of Mainutils class")
            return model
        except Exception as e:
            raise CustomException(e,sys) from e
    @staticmethod
    def get_model_params(self,model:object,x_train:DataFrame,y_train:DataFrame, model_config:dict)->Dict:
        logger.info("Entered the get_model_params method of Mainutils class")

        try:
            model_config = self.read_model_config_file()
            grid_search_config = model_config['grid_search']['params'] #cv, #verbose
            param_grid = model_config['search_param_grid']

            model_grid = GridSearchCV(
                params = param_grid,
                model = model,
                cv = grid_search_config.get('cv',3),
                verbose = grid_search_config.get('verbose',2),
                n_jobs = -1
            )

            model_grid.fit(x_train,y_train)

            logging.info("Exited the get_model_params method of MainUtils class")
            return model_grid.best_params_
        except Exception as e:
            raise CustomException(e,sys) from e

    @staticmethod
    def get_model_score(test_y:DataFrame,preds:DataFrame)-> float:
        logger.info("Entered the get_model_score method of MainUtils class")

        try:
            model_score = roc_auc_score(test_y,preds)
            logger.info(f"Model score is {model_score}")
            logger.info("Exited the get_model_score method of Mainutils class")

            return model_score
        except Exception as e:
            raise CustomException(e,sys) from e

    @staticmethod
    def save_object(file_path:str,obj:object) -> None:
        logger.info("Entered the save_object method of mainUtils class")

        try:
            with open(file_path,"wb") as file_obj:
                picle.dump(obj,file_obj)

            logger.info("Exited the save_object method of MainUtils class")

        except Exception as e:
            raise CustomException(e,sys) from e

    @staticmethod
    def get_best_model_with_name_and_score(model_list:list)-> Tuple[object,float]:
        logger.info(
            "Entered get_best_model_with_name_and_score method of MainUtils Class"
        )

        try:
            best_score = max(model_list)[0]
            best_model = max(model_list)[1]

            logger.info(
                "Exited the get_best_model_with_name_and_score method of MainUtils class"
            )

            return best_model,best_score
        except Exception as e:
            raise CustomException(e,sys) from e

    @staticmethod
    def load_object(file_path:str)-> object:
        logger.info("Entered the load_object method of MainUtils class")

        try:
            with open(file_path,"rb") as file_obj:
                obj = pickle.load(file_obj)

                logger.info("Exited the load_object method of MainUtils class")

                return obj

        except Exception as e:
            raise CustomException(e,sys) from e

    @staticmethod
    def unzip_file(filename:str,folder_name:str)-> None:
        logger.info('Entered the unzip_file method of MainUtils class')

        try:
            shutil.unpack_archive(filename,folder_name)
            logger.info("Exited the unzip_file method of MainUtils class")
        except Exception as e:
            raise CustomerException(e, sys) from e

         





