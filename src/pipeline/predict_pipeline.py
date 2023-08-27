import os
import sys

from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import logging
from src.request_model.diamond_features import DiamondFeatures
from src.utils import convert_diamond_features_to_dataframe, load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def predict(diamond_features: DiamondFeatures):
    try:
        logging.info('first check if the env variables are loaded, this also checks if the pickle files are created')
        if os.getenv('ARTIFACTS_FOLDER') == None:
            logging.info('Artifacts folder is not created')
            load_dotenv()
            initiate_preprocess_and_train()

        logging.info('convert features into dataframe')
        input_df = convert_diamond_features_to_dataframe(diamond_features)

        logging.info('load the preprocessor and model pickle file')
        preprocessor_path = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('PREPROCESSOR_PICKLE'))
        model_path = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('MODEL_PICKLE'))
        preprocessor_obj = load_object(preprocessor_path)
        model_obj = load_object(model_path)

        logging.info('Preprocess the input')
        input_df_transformed = preprocessor_obj.transform(input_df)

        logging.info('Predict the price')
        price_prediction = model_obj.predict(input_df_transformed)

        return price_prediction
    except Exception as e:
        raise CustomException(e, sys)

def initiate_preprocess_and_train():
    di = DataIngestion()
    train_data_path, test_data_path = di.initiate_data_ingestion()

    dt = DataTransformation()
    slim_X_train_transformed, slim_X_test_transformed, y_train, y_test, preprocessor_obj_file_path = dt.initiate_data_transformation(train_data_path, test_data_path)

    mt = ModelTrainer()
    test_score = mt.initiate_model_trainer(slim_X_train_transformed, slim_X_test_transformed, y_train, y_test)
