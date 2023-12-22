import os
import sys
from dataclasses import dataclass
from src.exception import CustomExceptionMessage
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "trained_model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("before reading model and the preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("finished reading model and the preprocessor")
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions

        except Exception as e:
            raise CustomExceptionMessage(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dat_frame(self):
        try:
            custom_data_imput_dic = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_imput_dic)

        except Exception as e:
            raise CustomExceptionMessage(e, sys)
