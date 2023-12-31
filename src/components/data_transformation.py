import sys
import os

import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class DataTransformationConfig:
    if os.getenv('ARTIFACTS_FOLDER') is None:
        load_dotenv()
    preprocessor_obj_file_path = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('PREPROCESSOR_PICKLE'))

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_columns = make_column_selector(dtype_exclude=['O', 'bool'])
            num_pipeline = Pipeline([
                ('median_imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
            ])
            logging.info('created numerical pipeline')

            color_pipeline = Pipeline([
                ('color_ord_encoder', OrdinalEncoder(categories=[['J', 'I', 'H', 'G', 'F', 'E', 'D']]))
            ])

            clarity_pipeline = Pipeline([
                ('clarity_ord_encoder', OrdinalEncoder(categories=[['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']]))
            ])

            logging.info('created categorical pipeline for color, clarity')

            column_transformer = ColumnTransformer([
                ('numerical_pipeline', num_pipeline, num_columns),
                ('color_pipeline', color_pipeline, ['color']),
                ('clarity_pipeline', clarity_pipeline, ['clarity'])
            ])
            logging.info('Created column_transformer')
            return column_transformer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Read train and test dataset into dataframe')

            logging.info('Get data transformer object')
            transformer_obj = self.get_data_transformer_object()
            target_column_name='price'

            logging.info('Split the features and target for train and test data')
            X_train = train_df.drop(columns=[target_column_name], axis='columns')
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis='columns')
            y_test = test_df[target_column_name]

            logging.info('Fit Transform the X_train')
            X_train_transformed = transformer_obj.fit_transform(X_train)

            logging.info('Transform X_test')
            X_test_transformed = transformer_obj.transform(X_test)

            logging.info('Create and save pickle file')
            save_object(self.data_transformation_config.preprocessor_obj_file_path, transformer_obj)

            return (
                X_train_transformed, X_test_transformed, y_train, y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
