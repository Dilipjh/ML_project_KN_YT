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
from sklearn.model_selection import GridSearchCV

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
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "LogisticRegression": LogisticRegression(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
            }

            params = {
                "Random Forest Regressor": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2', None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting Regressor": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost Regressor": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                },
                "LogisticRegression": {},
                "Linear Regression": {},
                "Ridge": {},
                "Lasso": {},
                "K neighbours Regressor": {
                    "n_neighbors": [5, 7, 9, 11],
                    # 'weights':['uniform','distance'],
                    # 'algorithm':['ball_tree','kd_tree','brute']
                },
                "Decision Tree Regressor": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
            }

            report = {}

            for i in range(len(list(models_list))):
                model_in_consideration = list(models_list.values())[i]
                """ if list(models_list.keys())[i] in list(params.keys()):
                    parameters_for_model = list(models_list.keys())[i]
                    logging.info("parameter values found")
                    print(
                        "parameter values found for {}".format(
                            list(models_list.keys())[i]
                        )
                    )
                else:
                    parameters_for_model = "no parameter"
                    logging.info(
                        "no parameter for {}".format(list(models_list.keys())[i])
                    )
                    print("no parameter for {}".format(list(models_list.keys()))[i]) """

                parameters_for_model = params[list(params.keys())[i]]

                gs = GridSearchCV(model_in_consideration, parameters_for_model, cv=3)
                gs.fit(X_train_features, Y_train)

                model_in_consideration.set_params(**gs.best_params_)
                model_in_consideration.fit(X_train_features, Y_train)

                Y_test_predicted = model_in_consideration.predict(X_test_features)
                mae, mse, r2_value = model_evaliation(
                    Y_predicted=Y_test_predicted, Y_test=Y_test
                )
                report[list(models_list.keys())[i]] = r2_value
                print("{0} model was run".format(model_in_consideration))
                logging.info("{0} model was run".format(model_in_consideration))

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
