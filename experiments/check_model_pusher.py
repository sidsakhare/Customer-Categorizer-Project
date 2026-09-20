import sys
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.components.model_pusher import ModelPusher


mp = ModelPusher(model_pusher_config= ModelPusherConfig, model_trainer_artifact=ModelTrainerArtifact )