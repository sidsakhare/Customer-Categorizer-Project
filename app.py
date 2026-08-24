from  src.logging import logger
import os
import sys
from src.exception import CustomException

try:
    1/0

except Exception as e:
    raise CustomException(e,sys) from e

logger.info("This is an info message")