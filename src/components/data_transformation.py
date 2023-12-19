import os
import sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomExceptionMessage

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        """This fucntion performs data transformation"""
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            logging.info("Numerical column scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("One hot encoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )

            logging.info("categorical column encoding completed")

            preprossor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    (" catagorical pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprossor

        except Exception as e:
            raise CustomExceptionMessage(e, sys)

    def initiate_data_tranformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("read training and trst df")
            logging.info("obtaining preprossor object")

            preprossor_object = self.get_data_transformer_obj()

            target_column = "math_score"
            numerical_columns = ["reading_score", "writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            traget_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            traget_feature_test_df = test_df[target_column]

            logging.info("applying preprosser object for test and tain dataframes")

            input_feature_train_arr = preprossor_object.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprossor_object.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(traget_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(traget_feature_test_df)]

            logging.info("saved prepocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprossor_object,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomExceptionMessage(e, sys)
