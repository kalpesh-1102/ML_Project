# Import necessary libraries
import os
import sys
from dataclasses import dataclass

# Importing different regression models from sklearn, catboost, and xgboost
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  # For evaluating the model performance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Import custom exception and logger for debugging and error handling
from src.exception import CustomException
from src.logger import logging

# Import utility functions (for saving the model and evaluating models)
from src.utils import save_object, evaluate_models

# Configuration class to define where to save the trained model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  # Path to save the final trained model

# Main class responsible for training and evaluating machine learning models
class ModelTrainer:
    def __init__(self):
        # Initialize the configuration
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Log the start of model training process
            logging.info("Splitting training and test input data")

            # Split features and target variable from training and test datasets
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # All rows, all columns except last (features)
                train_array[:, -1],   # All rows, last column (target)
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define a dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate all models and return a dictionary with model names and R2 scores
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models
            )

            # Get the best model score from the evaluation report
            best_model_score = max(sorted(model_report.values()))

            # Get the model name with the highest score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            # Fetch the best performing model using the name
            best_model = models[best_model_name]

            # Check if best model's performance is acceptable (R2 score threshold = 0.6)
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            # Log the model selected
            logging.info(f"Best found model on both training and testing dataset")

            # Save the best model as a .pkl file in the specified path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Use the best model to make predictions on the test set
            predicted = best_model.predict(X_test)

            # Calculate the R-squared score for the predictions
            r2_square = r2_score(y_test, predicted)

            # Return the R2 score as the model's evaluation result
            return r2_square

        except Exception as e:
            # If an error occurs during any step, raise a custom exception with details
            raise CustomException(e, sys)
