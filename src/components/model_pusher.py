import sys
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelPusherArtifact, ModelTrainerArtifact
from src.exception import CustomException
from src.logging import logger
from src.ml.model.local_estimator import LocalModelEstimator


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifact: ModelTrainerArtifact):

        self.model_trainer_artifact = model_trainer_artifact
        self.model_pusher_config = model_pusher_config
        self.estimator = LocalModelEstimator(
            model_path = model_pusher_config.model_file_path
        )
    
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        logger.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            logger.info("saving model to local folder")
            self.estimator.save_model(
                from_file= self.model_trainer_artifact.trained_model_file_path)

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path= self.model_pusher_config.model_file_path)

            logger.info("Saved model to local folder")
            logger.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logger.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys) from e