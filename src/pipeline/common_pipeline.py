from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

from dotenv import load_dotenv
import os

def check_if_model_is_trained():
    logging.info('first check if the env variables are loaded, this also checks if the pickle files are created')
    if not os.path.exists(os.getenv('ARTIFACTS_FOLDER')):
        logging.info('Artifacts folder is not created')
        load_dotenv()
        initiate_preprocess_and_train()


def initiate_preprocess_and_train():
    di = DataIngestion()
    train_data_path, test_data_path = di.initiate_data_ingestion()

    dt = DataTransformation()
    slim_X_train_transformed, slim_X_test_transformed, y_train, y_test, = dt.initiate_data_transformation(train_data_path, test_data_path)

    mt = ModelTrainer()
    test_score = mt.initiate_model_trainer(slim_X_train_transformed, slim_X_test_transformed, y_train, y_test)
