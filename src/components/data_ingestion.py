import os
import sys

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from dotenv import load_dotenv
from src.utils import convert_diamond_features_to_dataframe, load_object
from src.request_model.diamond_features import DiamondFeatures

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    if os.getenv('ARTIFACTS_FOLDER') is None:
        load_dotenv()
    train_data_path: str = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('TRAIN_CSV'))
    test_data_path: str = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('TEST_CSV'))
    raw_data_path: str = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('DATA_CSV'))

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion function')
        try:
            df = pd.read_csv('notebook/data/diamonds.csv')
            logging.info('Read dataset as dataframe')

            df = df.drop_duplicates()
            logging.info('Drop duplicates')

            df['x'] = df['x'].replace(0, np.nan)
            df['y'] = df['y'].replace(0, np.nan)
            df['z'] = df['z'].replace(0, np.nan)
            logging.info('Replace to 0 value in x, y, and z to nan')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            logging.info('created artifacts folder and saved the raw dataset')

            logging.info('Create slim df')
            slim_df = df[['carat', 'y', 'clarity', 'color', 'price']]

            logging.info('Train Test Split started')
            train_set, test_set = train_test_split(slim_df, test_size=0.3, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == '__main__':
    # di = DataIngestion()
    # train_data_path, test_data_path = di.initiate_data_ingestion()

    # dt = DataTransformation()
    # X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_obj_file_path = dt.initiate_data_transformation(train_data_path, test_data_path)

    # mt = ModelTrainer()
    # test_score = mt.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)
    # print(f'r2_score is: {test_score}')

    # preprocessor_path = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('PREPROCESSOR_PICKLE'))
    # model_path = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('MODEL_PICKLE'))
    # preprocessor_obj = load_object(preprocessor_path)
    # model_obj = load_object(model_path)

    # diamond_features = DiamondFeatures(carat=0.5, y=5.2, clarity="VS1", color="D")
    # print('123')
    # print(type(diamond_features))
    # print(diamond_features)

    # input_df = convert_diamond_features_to_dataframe(diamond_features)
    # print(input_df)
    # print()
    # input_df_transformed = preprocessor_obj.transform(input_df)
    # print(input_df_transformed)
    # print(type(input_df_transformed), input_df_transformed.shape)

    # price_prediction = model_obj.predict(input_df_transformed)
    # print('price_prediction:', price_prediction)
