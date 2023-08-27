import pandas as pd
import pickle

def get_slim_dataframe(transformer_obj, X_train_test_transformed):
    transformed_feature_names = []
    transformed_feature_names.extend(transformer_obj.transformers_[0][2])
    transformed_feature_names.extend(['cut', 'color', 'clarity'])
    X_train_test_transformed_df = pd.DataFrame(X_train_test_transformed, columns=transformed_feature_names)
    slim_X_transformed = X_train_test_transformed_df[['carat', 'y', 'clarity', 'color']]

    # train_test_transformed_df = pd.DataFrame(X_train_test_transformed, columns=transformed_feature_names)
    # train_test_transformed_df['price'] = y_train_test.values

    return slim_X_transformed

def save_object(file_path, obj):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
