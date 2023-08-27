import pandas as pd
import pickle
from src.request_model.diamond_features import DiamondFeatures

def save_object(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_object(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert_diamond_features_to_dataframe(diamond_features: DiamondFeatures):
    return pd.DataFrame([diamond_features.dict()])
