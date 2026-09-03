import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
from pandas import DataFrame
from sklearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.constants.training_pipeline import TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entitty.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.logging import logger
from src.entity.config_entity import SimpelImputerConfig
from src.utils.common import MainUtils


class DataTransformation:

    def __init__(self,
    data_ingetion_artifact:DataIngestionArtifact,
    data_validation_artifact: DataValidationArtifact,
    data_transformation_config: DataTransformationConfig):

            self.data_ingestion_artifact = data_ingetion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.imputer_config = SimpelImputerConfig()
            self.utils = MainUtils()


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
        test_set_with_new_featuures = pd.DataFrame()
        datasets = {"train_set": train_set, "test_set": test_set}

        for key in datasets:
            dataset = datasets[key]
            
            #Creating new field to store the age of customer
            dataset["AGE"] =pd.Timestamp.now().year - dataset['Year_Birth']

            # recoding the customers education level to numeric form (0: Basic, 1: Graduation, 2: Master, 3: PHD)
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            dataset["Education"] = encoder.fit_transform(dataset["Education"])
            print("Education Encoding")
            for index, class_name in enumerate(encoder.classes_):
                print(F"{index}:{class_name}")

            # recoding the customers marital status to numeric form (0:Absurd 1:Alone, 2:Divorced ,3:Married ,4:Single ,5:Together ,6:Widow ,7:YOLO)
            dataset["Marital_Status"] = encoder.fit_transform(dataset["Marital_Status"])
            print("-------------")
            print("Marital_Status Encoding")
            for index, class_name in enumerate(encoder.classes_):
                print(F"{index}:{class_name}")

            # creating a new field to store the number of children in the household
            dataset["Children"] = dataset["Kidhome"]+dataset["Teenhome"]

            # creating Family_Size
            dataset["Family_size"] = dataset["Children"] + dataset["Marital_Status"]+1

            #  creating a new field to store the total spending of the customer
            dataset['Total_Spending'] = dataset.iloc[:,[6,7,8,9,10,11]].sum(axis = 1)

            #  creating a new field how many promo done for customer
            promo_col = ['AcceptedCmp3','AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2']
            dataset.columns.get_indexer(promo_col)
            dataset['Total_Promo'] = dataset.iloc[:,[17,18,19,20,21]].sum(axis = 1)

            # The following code works out how long customer has been with the company 
            dataset["Days_as_Customer"] = (pd.Timestamp.now() - dataset["Dt_Customer"]).dt.days

            # Total number of promotions customer responced to 
            dataset['Offers_Responded_To'] = dataset.iloc[:,[17,18,19,20,21,23]].sum(axis = 1)

            # parental status of a customer
            dataset['parental_status'] = np.where(dataset['Children']>0,1,0)


            # dropping columns which are already used to create new features
            columns_to_drop = ['Year_Birth',"Kidhome","Teenhome"]
            dataset.drop(columns = columns_to_drop, axis = 1, inplace=True)
            dataset.rename(columns={"Marital_Status": "Marital Status","MntWines": "Wines","MntFruits":"Fruits",
                            "MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets",
                            "MntGoldProds":"Gold","NumWebPurchases": "Web","NumCatalogPurchases":"Catalog",
                            "NumStorePurchases":"Store","NumDealsPurchases":"Discount Purchases"},
                    inplace = True)

            dataset = dataset[["AGE","Education","Marital Status","parental_status","Children","Income","Total_Spending","Days_as_Customer","Recency","Wines","Fruits","Meat","Fish","Sweets","Gold","Web","Catalog","Store","Discount Purchases","Total_Promo","NumWebVisitsMonth"]]
            if key == 'train_set':
                train_test_with_new_features = pd.concat([train_test_with_new_features,dataset], axis = 0)
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
        outlier_features = ['Wines','Fruits','Meat','Fish','Sweets','Gold','AGE','Total_Spending']
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
        preprocessed_test_set = preprocessor.fit_transform(test_set)

        preprocessor_obj_dir = os.path.dirname(self.data_transformation_config.transformed_object_file_path)
        os.makedirs(preprocessor_obj_dir,exist_ok=True)
        self.utils.save_objects(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor)
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


                cluster_creater = ClusterCreaters()

                labelled_train_set = cluster_creater.initialize_clustering(preprocessed_data = preprocessed_train_set)
                labelled_test_set = cluster_creater.initialize_clustering(preprocessed_data = preprocessed_test_set)

                x_train = labelled_train_set.drop(columns = [TARGET_COLUMN], axis = 1)
                y_train = labelled_train_set[TARGET_COLUMN]

                x_test = labelled_test_set.drop(columns = [TARGET_COLUMN], axis = 1)
                y_test = labelled_test_set[TARGET_COLUMN]

                test_arr = np.c_[x_test,y_test]
                train_arr = np.c_[x_train,y_train]

                self.utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_file_path, array = train_arr)
                self.utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_file_path, array = test_arr)

                return data_transformation_artifact

            else:
                raise Exception("Data Validation is not successful. Please check the data validation artifact for more details")

        except Exception as e:
            raise CustomException(e,sys) from e
            

    

