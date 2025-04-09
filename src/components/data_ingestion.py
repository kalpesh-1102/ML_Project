# Importing necessary libraries
import os  # for file path and directory operations
import sys  # for accessing system-specific parameters and functions
from src.exception import CustomException  # custom exception handling
from src.logger import logging  # custom logger for logging messages
import pandas as pd  # for data manipulation and reading CSV files

from sklearn.model_selection import train_test_split  # to split the dataset into training and testing
from dataclasses import dataclass  # to create classes for storing configurations

# Configuration class for Data Ingestion using dataclass
@dataclass
class DataIngestionConfig:
    # These are the file paths where the data will be saved after ingestion
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to save testing data
    raw_data_path: str = os.path.join('artifacts', "raw.csv")  # Path to save raw/original data

# Main class that handles data ingestion
class DataIngestion:
    def __init__(self):
        # Creating an instance of DataIngestionConfig to access file paths
        self.ingestion_config = DataIngestionConfig()

    # Function to start the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")  # Logging entry into the method
        try:
            # Reading the raw dataset (in CSV format) using pandas
            df = pd.read_csv("notebook\\data\\stud.csv")  # <- Make sure this path is correct on your machine
            logging.info('Read the dataset as dataframe')  # Logging successful data read

            # Creating the artifacts directory if it doesn't already exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Saving the original raw dataset as 'raw.csv' inside artifacts folder
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")  # Logging train-test split starting

            # Splitting the data into training and testing sets (80% train, 20% test)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Saving training data to 'train.csv'
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            # Saving testing data to 'test.csv'
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")  # Logging completion of ingestion

            # Returning the paths of training and testing data
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        # If any error occurs during ingestion, it will be logged and raised using a custom exception
        except Exception as e:
            raise CustomException(e, sys)

# When this script is run directly, this block will execute
if __name__ == "__main__":
    obj = DataIngestion()  # Creating an object of DataIngestion class
    obj.initiate_data_ingestion()  # Calling the ingestion method to start the process
