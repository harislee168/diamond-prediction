from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import sys
from src.request_model.diamond_features import DiamondFeatures
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get('/')
async def hello_world():
    return {'message': 'hello_world'}

@app.post('/getprice')
async def get_price(diamond_features: DiamondFeatures):
    logging.info('Get price API is called')
    price_prediciton  = predict(diamond_features)
    logging.info('Turn np.float into float and round to the next 2 decimal places')
    return_price = round(float(price_prediciton[0]), 2)
    return {'price_prediciton': return_price}
