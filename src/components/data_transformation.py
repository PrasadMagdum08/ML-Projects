import os, sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import save_object

"""This line use to create a sequential pipeline to maintain the data in a one by one form."""

from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    
    """This class is use for transforming data into encoding form and convert it into scaler form for training and testing models."""

    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:

            """This are the numerical features in the dataset."""
            numerical_features = ['writing_score', 'reading_score']

            """These are the categorical feature in the dataset."""
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoded', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Numerical feature scaling conpleted.')
            logging.info('Categorical feature encoding conpleted.')

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data, test_data):
        
        
        try:

            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info('Read train and test data completed.')

            preprecessor_obj = self.get_transformer_object()

            target_feature_name = 'math_score'

            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_feature_name], axis=1)
            target_feature_train_df = train_df[target_feature_name]

            input_feature_test_df = test_df.drop(columns=[target_feature_name],axis=1)
            target_feature_test_df = test_df[target_feature_name]

            logging.info('Applying preprocessing object on training and testing dataframe.')

            input_feature_train_arr = preprecessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprecessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocessing object.')

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path,
                obj = preprecessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
 


        

