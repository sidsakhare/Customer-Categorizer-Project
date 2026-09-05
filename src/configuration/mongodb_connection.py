import os
import sys

import certifi
import pymongo

from src.constants.database import DATABASE_NAME
from src.constants.env_variable import MONGODB_URL_KEY
from src.exception import CustomException

ca = certifi.where()

class MongoDBClient():
    client = None

    def __init__(self,database_name = DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                mongodb_url = os.getenv(MONGODB_URL_KEY)
                if mongodb_url is None:
                    raise Exception(f"Enviroment key:{MONGODB_URL_KEY} is not set")
                if mongodb_url.startswith("mongodb+srv://"):
                    # Atlas / cloud — needs TLS
                    MongoDBClient.client = pymongo.MongoClient(mongodb_url, tlsCAFile=ca)
                else:
                    # Local MongoDB — no TLS
                    MongoDBClient.client = pymongo.MongoClient(mongodb_url)

            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise CustomException(e,sys) from e