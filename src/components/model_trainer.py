import os
import sys

from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, slim_X_train_transformed,
                               slim_X_test_transformed, y_train_df, y_test_df):
        try:
            logging.info('initializa the params for XGBoost')
            params = {
                'gamma': 1.208228629767141,
                'learning_rate': 0.06038269184903254,
                'max_depth': 5,
                'min_child_weight': 2.3791345006085356,
                'n_estimators': 218,
                'subsample': 0.9518781280875572
            }

            logging.info('create and fit the model with train_df')
            xgb_model = XGBRegressor(**params)
            xgb_model.fit(slim_X_train_transformed, y_train_df)

            logging.info('predict the test data')
            y_pred = xgb_model.predict(slim_X_test_transformed)

            logging.info('check the prediciton score of the model')
            test_score = r2_score(y_true=y_test_df, y_pred=y_pred)
            logging.info(f'prediction score of the model is {test_score}')

            save_object(
                self.model_trainer_config.trained_model_file_path,
                xgb_model
            )
            return test_score
        except Exception as e:
            raise CustomException(e, sys)
