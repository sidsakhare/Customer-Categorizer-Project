import logging
import os

from src.constants.training_pipeline import LOG_DIR,LOG_FILE


logs_path = os.path.join(LOG_DIR,LOG_FILE)
LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE)
os.makedirs(LOG_DIR,exist_ok=True)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s: %(name)s : %(levelname)s : %(message)s]",
    level=logging.INFO)

logger = logging.getLogger("customer_segmentation_logger")