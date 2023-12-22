import os, sys
import pandas as pd
import numpy as np
import dill
from src.exception import CustomExceptionMessage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def save_object(file_path, object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(object, file_obj)

    except Exception as e:
        raise CustomExceptionMessage(e, sys)


def model_evaliation(Y_predicted, Y_test):
    mae = mean_absolute_error(Y_predicted, Y_test)
    mse = mean_squared_error(Y_predicted, Y_test)
    r2_value = r2_score(Y_predicted, Y_test)

    return (mae, mse, r2_value)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomExceptionMessage(e, sys)
