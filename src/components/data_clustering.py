import sys
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import PCAconfig
from src.exception import CustomException
from src.logging import logger


class CreatClusters:
    def __init__(self):
        self.pca_config = PCAconfig()

    def get_dataset_using_pca(self,preprocessed_data:DataFrame):
        '''
        Method Name :   get_dataset_using_pca
        Description :   This method applies PCA over the preprocessed dataset.
            
        Output      :   pca object is created and preprocessed dataset is fitted and returned 
        On Failure  :   Write an exception log and then raise an exception
            
        Version     :   0.0

        '''

        try:
            logger.info("initializing PCA...")
            reduced_dataset = PCA(**self.pca_config.__dict__).fit(preprocessed_data)

            logger.info("PCA transformation is done")
            return reduced_dataset
        except Exception as e:
            raise CustomException(e,sys) from e

    def initialize_clustering(self, preprocessed_data:DataFrame) -> DataFrame:
        '''
         Method Name :   initialize_clustering
        Description :   This method initiates the clustering process 
        
        Output      :   Data is clustered and the cluster names are used as lables to the preprocessed data and is returned.
        On Failure  :   Write an exception log and then raise an exception
        
        Version     :   0.1
        '''

        try:
            logger.info("initializing clustering...")

            reduced_data = self.get_dataset_using_pca(preprocessed_data)

            model = KMeans(n_clusters = 3).fit_predict(preprocessed_data)

            preprocessed_data[TARGET_COLUMN] = model.labels_.astype(int)
            logging.info("Clustering is done")
            return preprocessed_data

        except Exception as e:
            raise CustomException(e,sys) from e