# Importing necessary libraries
import os  # For interacting with the operating system (e.g., creating directories)
import sys  # To access system-specific parameters and functions
import pandas as pd  # For data manipulation and analysis using DataFrames
from dataclasses import dataclass  # To simplify class definitions for storing config
from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets

# Importing custom modules for exception handling, logging, and data transformation
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

# Configuration class using @dataclass to hold file paths for saving data
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")    # Path to save test data
    raw_data_path: str = os.path.join('artifacts', "raw.csv")      # Path to save raw/original data

# Main class responsible for ingesting (loading and saving) data
class DataIngestion:
    def __init__(self):
        # Create an instance of the DataIngestionConfig to access file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process...")

        try:
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv("notebook\\data\\stud.csv")
            logging.info("Dataset loaded into dataframe.")

            # Ensure the 'artifacts' directory exists before saving files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset as 'raw.csv'
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the dataset into training and test sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed.")

            # Save the training set to 'train.csv'
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Save the test set to 'test.csv'
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully.")

            # Return the paths to the saved train and test data
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            # If any exception occurs, raise it as a CustomException
            raise CustomException(e, sys)

# Entry point when running this file directly (not importing as a module)
if __name__ == "__main__":
    # Create an object of the DataIngestion class
    obj = DataIngestion()

    # Start the ingestion process and get paths to train/test datasets
    train_data, test_data = obj.initiate_data_ingestion()

    # Create an object of DataTransformation class
    data_transformation = DataTransformation()

    # Call the transformation method using the paths returned above
    data_transformation.initiate_data_transformation(train_data, test_data)
