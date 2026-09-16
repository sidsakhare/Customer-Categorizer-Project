import os
import sys
import shutil
import pickle
from pandas import DataFrame

from src.exception import CustomException
from src.ml.model.estimator import CustomerSegmentationModel

class LocalModelEstimator:
    '''
    saves/loads/predict using a CustomerSegmentationModel stored on local file
    system
    '''
    def __init__(self,model_path:str):
        self.model_path = model_path
        self.loaded_model = CustomerSegmentationModel = None

    def is_model_present(self) -> bool:
        return os.path.exist(self.model_path)

    def load_model(self) -> CustomerSegmentationModel:
        return pickle.load(self.model_path)

    def save_model(self, from_file: str, remove : bool = False) -> None:
        try:
            os.makedirs(os.path.dirname(self.model_path),exist_ok= True)
            if remove:
                os.replace(from_file, self.model_path)
            else:
                shutil.copy(from_file, self.model_path)
        except Exception as e:
            raise CustomException(e,sys) from e

    def predict(self,dataframe: DataFrame):

        try:
            if self.loaded_model is None:
                self.loaded_model = load_model()
            
            return self.loaded_model.predict(dataframe)

        except Exception as e:
            raise CustomException(e,sys) from e