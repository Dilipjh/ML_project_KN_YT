import os
import sys
import pandas as pd

from src.exception import CustomExceptionMessage
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainingConfig

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionCofig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionCofig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method")
        try:
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("read the data as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            logging.info("creating test data split")
            train_set, test_set = train_test_split(df, random_state=42, test_size=0.2)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("split data to train and test")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomExceptionMessage(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_tranformation(
        train_data_path, test_data_path
    )

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(
        train_arr=train_array,
        test_arr=test_array,
    )
