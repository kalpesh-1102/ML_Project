# Importing required libraries
import sys  # For system-specific error details
from src.logger import logging  # Using the custom logger you created

# Function to extract detailed error message
def error_message_detail(error, error_detail: sys):
    """
    This function captures the full error details such as:
    - File name where the error occurred
    - Line number of the error
    - Error message

    It helps in debugging by giving clear context.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract exception traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename where error occurred
    error_message = "Error occurred in Python script: [{0}] at line [{1}] with message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message  # Return the full formatted error message

# Creating a custom exception class
class CustomException(Exception):
    """
    Custom exception class that extends the base Exception class.
    It uses the above error_message_detail function to generate a useful error message.
    """

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)  # Initialize parent Exception class
        # Store the formatted error message in the object
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        # Return the detailed error message when the exception is printed
        return self.error_message


# --- Optional test code to check if your custom exception is working correctly ---

# if __name__ == "__main__":
#     try:
#         a = 1 / 0  # This will cause a divide-by-zero error
#     except Exception as e:
#         logging.info("Divide by Zero")  # Log the issue
#         raise CustomException(e, sys)  # Raise your custom exception with full error info
