import pandas as pd
import pickle

def reconstruct_to_dataframe(transformer_obj, X_train_test_transformed):
    transformed_feature_names = []
    transformed_feature_names.extend(transformer_obj.transformers_[0][2])
    transformed_feature_names.extend(['color', 'clarity'])
    slim_df = pd.DataFrame(X_train_test_transformed, columns=transformed_feature_names)
    return slim_df

def save_object(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
