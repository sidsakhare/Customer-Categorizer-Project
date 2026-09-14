import sys
from typing import Tuple
from pandas import DataFrame

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.components.model_pusher import ModelPusher

from src.exception import CustomException
from src.logging import logger
from src.entity import artifact_entity(
                                        DataIngestionArtifact,
                                        DataTransformationArtifact,
                                        DataValidationArtifact,
                                        ModelEvaluationArtifact,
                                        ModelTrainerArtifact)

from src.entity.config_entity import (DataIngestionConfig,
                                        DataTransformationConfig,
                                        DataValidationConfig,
                                        ModelEvaluationConfig,
                                        ModelPusherConfig,
                                        ModelTrainerConfig)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_confiig = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher = ModelPusherConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Entered start_data_ingestion method of TrainPipeline class")

        try:
            logger.info("Getting data from mongodb")

            data_ingestion = DataIngestion(data_ingestion_config= self.data_ingestion_config)
            data_ingestion-artifact = data_ingestion.initial_data_ingestion()
            logging.info("Got the train_set and test_set from mongodb")

            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e


    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        logger.info("Entered the  start_data_validation mrthod of TrainPipeline class")

        try:
            data_validation = DataValidation(
                data_validation_config= self.data_validation_config
                data_ingestion_artifact = data_ingestion_artifact
            )

            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info("Performed the data validation operation")
            logger.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact

        except Exception as e:
            raise CustomerException(e, sys) from e

    def start_data_transformation(self,
                                    data_ingestion_artifact: DataIngestionArtifact,
                                    data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:

        logger.info("Entered start_data_transformation method of TrainPipeline class") 

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact = data_ingestion_artifact,
                data_validation_artifact= data_validation_artifact,
                data_transformation_config= self.data_transformation_config
            )

            data_transformation_artifact = data_transformation.initial_data_transformation()

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys) from e


    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:

        logger.info("Entered start_model_trainer method of TrainPipeline class")

        try:
            model_trainer = ModelTrainer(data_transformation_artifact = data_transformation_artifact,
                                        model_trainer_config = self.model_trainer_confiig)

            
            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e,sys) from e


    def start_model_evaluation(
        self,
        data_ingestion_artifact : DataIngestionArtifact,
        model_trainer_artifact : ModelTrainerArtifact,
        data_transformation_artifact : DataTransformationArtifact
    )-> ModelEvaluationArtifact:

        try:
            model_evaluation = ModelEvaluation(
                model_eval_config= self.model_evaluation_config,
                data_ingestion_artifact= data_ingestion_artifact,
                model_trainer_artifact= model_trainer_artifact,
                data_transformation_artifact= data_transformation_artifact
            )

            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e,sys)

        
    def start_model_pusher(self,model_trainer_artifact:ModelTrainerArtifact):
        try:
            model_pusher = ModelPusher(
                model_trainer_artifact= model_trainer_artifact,
                model_pusher_config= self.model_evaluation_config
            )

            model_pusher_artifact = model_pusher.initiate_model_pusher()

            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e,sys) from e

    def run_pipeline(self)-> None:
        logger.info("Entered the run_pipeline method of TrainPipeline class ")

        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact= data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact= data_ingestion_artifact,
                data_validation_artifact= data_validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact= data_transformation_artifact
            ) 
            logger.info("Model Trained Succesfully")

            model_evaluation_artifact = self.start_model_evaluation(
                data_ingestion_artifact= data_ingestion_artifact,
                model_trainer_artifact= model_trainer_artifact,
                data_transformation_artifact= data_transformation_artifact
            )
            if not model_trainer_artifact.is_model_accepted:
                logger.info(f"Model not accepted")
                return None
            model_pusher_artifact = self.start_model_pusher(
                model_trainer_artifact= model_trainer_artifact
            )
            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise CustomerException(e, sys) from e



        

    

        
