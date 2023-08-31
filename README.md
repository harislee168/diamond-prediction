## End to end machine learning project predicting the price of diamond

### After you clone, you need to create .env file in the root directory to be able to use run the model. Below are the thing you need to provide in the .env

ARTIFACTS_FOLDER=artifacts
TRAIN_CSV=train.csv
TEST_CSV=test.csv
DATA_CSV=data.csv
PREPROCESSOR_PICKLE=preprocessor.pkl
MODEL_PICKLE=model.pkl
ALLOW_ORIGINS= YOUR_UI_URL for example: http://localhost:3000

Dataset url: https://www.kaggle.com/datasets/amirhosseinmirzaie/diamonds-price-dataset

To run type uvicorn src.api.fast:app --host 0.0.0.0 --port 8080

Open your browser you can access it from http://localhost:8080/docs
