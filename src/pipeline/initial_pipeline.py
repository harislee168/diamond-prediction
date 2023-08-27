import sys
import os

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.pipeline.common_pipeline import check_if_model_is_trained
from src.response_model.diamond_initial_value import DiamondInitialValue

def get_initial_values():
    try:
        #carat, y, clarity, color
        check_if_model_is_trained()

        logging.info('Load the raw dataframe')
        raw_data_path: str = os.path.join(os.getenv('ARTIFACTS_FOLDER'), os.getenv('DATA_CSV'))
        raw_df = pd.read_csv(raw_data_path)

        min_carat = float(raw_df['carat'].min())
        max_carat = float(raw_df['carat'].max())

        min_y = float(raw_df['y'].min())
        max_y = float(raw_df['y'].max())

        #Hard code the clarity and color to line up from the worst to best
        clarities = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
        colors = ['J', 'I', 'H', 'G', 'F', 'E', 'D']

        logging.info('Construct the diamond initial value object')
        diamond_initial = DiamondInitialValue(
            min_carat= min_carat,
            max_carat= max_carat,
            min_y= min_y,
            max_y= max_y,
            clarities= clarities,
            colors= colors)

        return diamond_initial
    except Exception as e:
        raise CustomException(e, sys)
