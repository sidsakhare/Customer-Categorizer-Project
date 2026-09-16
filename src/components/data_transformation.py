import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.common import load_numpy_array,save_numpy_array_data, write_yaml_file
from sklearn.impute import SimpleImputer

from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.logging import logger
from src.entity.config_entity import SimpleImputerConfig
from src.utils.common import Mainutils
from src.components import data_clustering


class DataTransformation:

    def __init__(self,
    data_ingetion_artifact:DataIngestionArtifact,
    data_validation_artifact: DataValidationArtifact,
    data_transformation_config: DataTransformationConfig):

            self.data_ingestion_artifact = data_ingetion_artifact
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

    def get_new_features(self, train_set: DataFrame, test_set: DataFrame) -> DataFrame:
        '''
        method : get_new_features
        description : This method is used to create new features for the training and testing sets
        version : 1.0
        '''

        train_set_with_new_features = pd.DataFrame()
        test_set_with_new_features = pd.DataFrame()
        datasets = {"train_set": train_set, "test_set": test_set}

        for key in datasets:
            dataset = datasets[key]

            # Converting Birth_Year and DT_Customer features to datetime 
            REFERANCE_DATE = pd.Timestamp('2026-01-01')
            dataset["Dt_Customer"]  = pd.to_datetime(dataset["Dt_Customer"],format='mixed')


            #Creating new field to store the age of customer
            dataset["Age"] = REFERANCE_DATE.year - dataset['Year_Birth']

            # recoding the customers education level to numeric form (0: Basic, 1: Graduation, 2: Master, 3: PHD)
            education_mapping = { "Basic": 0,'2n Cycle':1,"Graduation": 1,"Master": 2, "PhD": 3 }
            dataset["Education"] = dataset["Education"].map(education_mapping)

            # recoding the customers marital status to numeric form (0:Absurd 0:Alone, 0:Divorced ,1:Married ,0:Single ,1:Together ,0:Widow ,0:YOLO)
            dataset['Marital_Status'].replace({"Married":1, "Together":1, "Absurd":0, "Widow":0, "YOLO":0, "Divorced":0, "Single":0,"Alone":0},inplace=True) 

            # creating a new field to store the number of children in the household
            dataset["Children"] = dataset["Kidhome"]+dataset["Teenhome"]

            # creating Family_Size
            dataset["Family_size"] = dataset["Children"] + dataset["Marital_Status"]+1

            #  creating a new field to store the total spending of the customer
            spending_columns = [
                                "MntWines",
                                "MntFruits",
                                "MntMeatProducts",
                                "MntFishProducts",
                                "MntSweetProducts",
                                "MntGoldProds"
                                                    ]

            dataset["Total_Spending"] = dataset[spending_columns].sum(axis=1)

            #  creating a new field how many promo done for customer
            promo_columns = [
                                "AcceptedCmp1",
                                "AcceptedCmp2",
                                "AcceptedCmp3",
                                "AcceptedCmp4",
                                "AcceptedCmp5"
                            ]

            dataset["Total_Promo"] = dataset[promo_columns].sum(axis=1)

            # The following code works out how long customer has been with the company 
            dataset["Days_as_Customer"] = (pd.Timestamp.now() - dataset["Dt_Customer"]).dt.days

            # Total number of promotions customer responced to 
            dataset['Offers_Responded_To'] = dataset.iloc[:,[17,18,19,20,21,23]].sum(axis = 1)

            # parental status of a customer
            dataset['Parental_Status'] = np.where(dataset['Children']>0,1,0)


            # dropping columns which are already used to create new features
            columns_to_drop = ['Year_Birth',"Kidhome","Teenhome"]
            dataset.drop(columns = columns_to_drop, axis = 1, inplace=True)
            dataset.rename(columns={"Marital_Status": "Marital Status","MntWines": "Wines","MntFruits":"Fruits",
                            "MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets",
                            "MntGoldProds":"Gold","NumWebPurchases": "Web","NumCatalogPurchases":"Catalog",
                            "NumStorePurchases":"Store","NumDealsPurchases":"Discount_Purchases"},
                    inplace = True)

            dataset = dataset[["Age","Education","Marital Status","Parental_Status","Children","Income","Total_Spending","Days_as_Customer","Recency","Wines","Fruits","Meat","Fish","Sweets","Gold","Web","Catalog","Store","Discount_Purchases","Total_Promo","NumWebVisitsMonth"]]
            if key == 'train_set':
                train_set_with_new_features = pd.concat([train_set_with_new_features,dataset], axis = 0)
            else:
                test_set_with_new_features = pd.concat([test_set_with_new_features,dataset], axis = 0)

        logger.info("New features has been created successfully for both training and testing sets")
        return train_set_with_new_features, test_set_with_new_features


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
            "Imputer", SimpleImputer(strategy='constant',fill_value=0))
        ,(
        "Standardscaler", StandardScaler())])

        outlier_features_pipeline = Pipeline(steps=[("imputer",SimpleImputer(strategy="constant",fill_value=0)),
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

                labelled_train_set = cluster_creater.initialize_clustering(preprocessed_data = preprocessed_train_set)
                labelled_test_set = cluster_creater.initialize_clustering(preprocessed_data = preprocessed_test_set)

                x_train = labelled_train_set.drop(columns = [TARGET_COLUMN], axis = 1)
                y_train = labelled_train_set[TARGET_COLUMN]

                x_test = labelled_test_set.drop(columns = [TARGET_COLUMN], axis = 1)
                y_test = labelled_test_set[TARGET_COLUMN]

                test_arr = np.c_[x_test,y_test]
                train_arr = np.c_[x_train,y_train]

                save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_file_path, array = train_arr)
                save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_file_path, array = test_arr)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=transformed_object_file_path,
                    transformed_train_file_path=transformed_train_file_path,
                    transformed_test_file_path=transformed_test_file_path
                )


                return data_transformation_artifact

            else:
                raise Exception("Data Validation is not successful. Please check the data validation artifact for more details")

        except Exception as e:
            raise CustomException(e,sys) from e
    

