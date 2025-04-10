# Importing essential libraries
import os  # To handle file paths
import sys  # For system-specific operations (used in exception handling)
from dataclasses import dataclass  # To define simple configuration classes

# Data manipulation and preprocessing
import numpy as np  # For numerical array operations
import pandas as pd  # For working with DataFrames

# Scikit-learn tools for preprocessing
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # For scaling and encoding
from sklearn.pipeline import Pipeline  # For chaining preprocessing steps
from sklearn.compose import ColumnTransformer  # For applying different pipelines to columns

# Custom modules for exception handling, logging, and saving files
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # Custom function to save objects (e.g., pickle files)

# Configuration class to store the path where the preprocessor object will be saved
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

# Main class that handles all transformation logic
class DataTransformation:
    def __init__(self):
        # Initialize configuration
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Creates preprocessing pipelines for both numerical and categorical features.
        Returns a ColumnTransformer object combining both pipelines.
        '''
        try:
            # Defining numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender", "race_ethnicity",
                "parental_level_of_education", "lunch",
                "test_preparation_course"
            ]

            # Pipeline for numerical features:
            # 1. Handle missing values using median
            # 2. Standardize the features (mean=0, std=1)
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features:
            # 1. Fill missing values with the most frequent value
            # 2. Convert categories to one-hot vectors
            # 3. Scale without centering (because data is sparse)
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor  # Return the complete preprocessor

        except Exception as e:
            # Raise a custom exception with error info and system context
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Applies the preprocessing steps to train and test datasets.
        Returns transformed arrays and the path to the saved preprocessor.
        '''
        try:
            # Load the train and test data from CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test data loaded.")

            # Get the preprocessing object (ColumnTransformer)
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target/output column
            target_column_name = "math_score"

            # Separate input features (X) and target variable (y) for train and test sets
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Fit the preprocessor on training data and transform both train and test
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target variable into a final NumPy array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object (e.g., for later use during model prediction)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed and object saved.")

            # Return transformed train/test data and the path to the saved preprocessor
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
