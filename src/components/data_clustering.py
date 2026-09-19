import sys
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np



from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import PCAconfig
from src.exception import CustomException
from src.logging import logger


class CreatClusters:
    def __init__(self):
        self.pca_config = PCAconfig()
        self.pca = None
        self.Kmeans = None


    def fit_clusters(self,train_data: np.ndarray) -> np.ndarray:
        'Fit PCA + KMeans on train data only, return train cluster labels.'

        try:
            logger.info("Fitting PCA and KMeans on train data")
            self.pca = PCA(**self.pca_config.__dict__)
            reduced_data = self.pca.fit_transform(train_data)

            self.Kmeans = KMeans(n_clusters= 3,n_init = 10,random_state=42)
            return self.Kmeans.fit_predict(reduced_data).astype(int)

        except Exception as e:
            raise CustomException(e,sys) from e

    def predict_clusters(self,data:np.ndarray) -> np.ndarray:
        "Assign the clusters using already fitted KMeans and PCA"
        try:
            if self.pca is None or self.Kmeans is None:
                raise ValueError("Call fit clusters on train data first")

            return self.Kmeans.predict(self.pca.transform(data)).astype(int)
        except Exception as e:
            raise CustomException(e,sys) from e


