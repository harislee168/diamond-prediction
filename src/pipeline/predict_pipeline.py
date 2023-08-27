import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.request_model.diamond_features import DiamondFeatures
from src.utils import convert_diamond_features_to_dataframe, load_object
from src.pipeline.common_pipeline import check_if_model_is_trained

def predict(diamond_features: DiamondFeatures):
    try:
        check_if_model_is_trained()
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
