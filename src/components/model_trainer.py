import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomExceptionMessage
from src.logger import logging
from src.utils import model_evaliation, save_object


@dataclass
class ModelTrainingConfig:
    model_trained_path = os.path.join("artifacts", "trained_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train_features, Y_train, X_test_features, Y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            models_list = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LogisticRegression": LogisticRegression(),
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),
            }

            report = {}

            for i in range(len(list(models_list))):
                model_in_consideration = list(models_list.values())[i]
                model_in_consideration.fit(X_train_features, Y_train)
                Y_test_predicted = model_in_consideration.predict(X_test_features)
                mae, mse, r2_value = model_evaliation(
                    Y_predicted=Y_test_predicted, Y_test=Y_test
                )
                report[list(models_list.keys())[i]] = r2_value

            best_score = max(sorted(report.values()))
            best_model_fun_name = list(report.keys())[
                list(report.values()).index(best_score)
            ]
            best_model = models_list[best_model_fun_name]
            logging.info("best model found")

            if best_score < 0.6:
                raise CustomExceptionMessage(
                    "the best model is less than 60 percentage accurate"
                )

            print(report)

            print(
                "The best model is:{0} with a r2 score:{1}".format(
                    best_model, best_score
                )
            )

            save_object(
                file_path=self.model_trainer_config.model_trained_path,
                object=best_model,
            )

        except Exception as e:
            raise CustomExceptionMessage(e, sys)
