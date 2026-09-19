import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.common import save_numpy_array_data
from sklearn.impute import SimpleImputer
from typing import Tuple

from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.logging import logger
from src.entity.config_entity import SimpleImputerConfig
from src.utils.common import Mainutils
from src.components.data_clustering import CreatClusters


REFERENCE_DATE = pd.Timestamp("2026-01-01")
EDUCATION_MAP = {"Basic": 0, "2n Cycle": 1, "Graduation": 1, "Master": 2, "PhD": 3}
FINAL_COLUMNS = ["Age", "Education", "Marital Status", "Parental_Status", "Children", "Income",
                        "Total_Spending", "Days_as_Customer", "Recency", "Wines", "Fruits", "Meat", "Fish",
                        "Sweets", "Gold", "Web", "Catalog", "Store", "Discount_Purchases", "Total_Promo",
                        "NumWebVisitsMonth"]
MAX_AGE = 96          # Year_Birth >= 1930 with a 2026 reference date
MAX_INCOME = 200_000  # drops the 666,666 outlier; the test-set max is about 160,000
class DataTransformation:

    def __init__(self,
    data_ingestion_artifact:DataIngestionArtifact,
    data_validation_artifact: DataValidationArtifact,
    data_transformation_config: DataTransformationConfig):

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.imputer_config = SimpleImputerConfig()
            self.utils = Mainutils()



    @staticmethod
    def read_data(file_path:str) ->DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys) from e

    @staticmethod
    def engineer_features(df: DataFrame, remove_outliers: bool = True) -> DataFrame:
        df = df.copy()
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
        df["Age"] = REFERENCE_DATE.year - df["Year_Birth"]
        df["Education"] = df["Education"].map(EDUCATION_MAP)
        df["Marital Status"] = df["Marital_Status"].isin(["Married", "Together"]).astype(int)
        df["Children"] = df["Kidhome"] + df["Teenhome"]
        df["Parental_Status"] = (df["Children"] > 0).astype(int)
        df["Total_Spending"] = df[["MntWines", "MntFruits", "MntMeatProducts",
                                "MntFishProducts", "MntSweetProducts", "MntGoldProds"]].sum(axis=1)
        df["Total_Promo"] = df[[f"AcceptedCmp{i}" for i in range(1, 6)]].sum(axis=1)
        df["Days_as_Customer"] = (REFERENCE_DATE - df["Dt_Customer"]).dt.days

        if remove_outliers:
            rows_before = len(df)
            keep = (df["Age"] <= MAX_AGE) & (df["Income"].isna() | (df["Income"] <= MAX_INCOME))
            df = df[keep]
            logger.info(f"Removed {rows_before - len(df)} outlier rows (Age > {MAX_AGE} or Income > {MAX_INCOME})")

        df = df.rename(columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat",
                                "MntFishProducts": "Fish", "MntSweetProducts": "Sweets",
                                "MntGoldProds": "Gold", "NumWebPurchases": "Web",
                                "NumCatalogPurchases": "Catalog", "NumStorePurchases": "Store",
                                "NumDealsPurchases": "Discount_Purchases"})
        return df[FINAL_COLUMNS].reset_index(drop=True)

    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> Tuple[DataFrame, DataFrame]:
        train_out = self.engineer_features(train_set)
        test_out = self.engineer_features(test_set)
        logger.info("New features created for train and test sets")
        return train_out, test_out


    def transform_data(self,train_set: DataFrame, test_set: DataFrame)-> DataFrame:
        '''
        method : transform_data
        description : This method is used to transform the training and testing sets
        version : 1.0
        '''

        logger.info("Starting data transformation process")
        numeric_features = [feature for feature in train_set.columns if train_set[feature].dtype != "O"]
        outlier_features = ['Wines','Fruits','Meat','Fish','Sweets','Gold','Age','Total_Spending']
        numeric_features = [x for x in numeric_features if x not in outlier_features]

        logger.info("Initializing the StandardScaler and SimpleImputer for numeric features")

        numeric_pipeline = Pipeline(steps =[(
            "Imputer", SimpleImputer(strategy='median',fill_value=0))
        ,(
        "Standardscaler", StandardScaler())])

        outlier_features_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="median",fill_value=0)),
        ("Transformer",PowerTransformer(standardize=True))])

        preprocessor = ColumnTransformer(
            [
                ("Numeric_pipeline",numeric_pipeline,numeric_features),
                ("outlier features pipeline",outlier_features_pipeline,outlier_features)
            ]
        )

        preprocessed_train_set = preprocessor.fit_transform(train_set)
        preprocessed_test_set = preprocessor.transform(test_set)

        preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
        os.makedirs(preprocessor_obj_dir,exist_ok=True)
        self.utils.save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor)
        logger.info("Exited the transform_data method of DataTransformation class")

        return preprocessed_train_set, preprocessed_test_set

    
    def initial_data_transformation(self):
        '''
        method : initial_data_transformation
        description : This method is used to perform initial data transformation on the component of pipeline

        output: data transformation object is created and returned
        failure: raises exception if any error occurs

        version : 1.0
        '''
        logger.info("Entered the initial_data_transformation method of DataTransformation class")

        try:
            if self.data_validation_artifact.validation_status:
                train_set = DataTransformation.read_data(file_path = self.data_ingestion_artifact.trained_file_path)
                test_set = DataTransformation.read_data(file_path = self.data_ingestion_artifact.test_file_path)

                train_set, test_set = self.get_new_features(train_set = train_set, test_set = test_set)

                logger.info("Got the processor object")

                preprocessed_train_set, preprocessed_test_set = self.transform_data(train_set = train_set, test_set = test_set)

                cluster_creater = CreatClusters()

                y_train = cluster_creater.fit_clusters(preprocessed_train_set)
                y_test = cluster_creater.predict_clusters(preprocessed_test_set)

                # Features stay as the scaled data; PCA is only used to find the clusters
                train_arr = np.c_[preprocessed_train_set, y_train]
                test_arr = np.c_[preprocessed_test_set, y_test]

                save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_file_path, array = train_arr)
                save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_file_path, array = test_arr)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                )


                return data_transformation_artifact

            else:
                raise Exception("Data Validation is not successful. Please check the data validation artifact for more details")

        except Exception as e:
            raise CustomException(e,sys) from e
    

