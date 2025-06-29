import sys,os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,RobustScaler,MinMaxScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

from customerchurn.constant.training_pipeline import TARGET_COLUMN
#from customerchurn.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from customerchurn.entity.artifact_entity import DataTransformationArtifact,DataValidationArtifact
from customerchurn.entity.config_entity import DataTransformationConfig
from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging
from customerchurn.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig
                 ):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
            
        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def get_data_transformer_object(self,num_columns:list,cat_columns:list)->Pipeline:
        '''
        this fuction will preform following preprocessing steps 
        01. handling missing values - use SimpleImputer with strategry - most_frequenct
        02. Encording - labelencoder 
        03. Scaling - Use standardScaler,robustscaler and minmax 
        
        
        '''
        logging.info("Entered get_data_transformation_object method of Transformation class")
        
        try:
            
            #seperate numerical and categorical columns 
            
            
            #Pipeline
            numeric_transformer = Pipeline(
                steps=[
                  ('imputer',SimpleImputer(strategy='median')),
                  ('scaler',StandardScaler())  
                    
                ])
                
            categorical_transformer = Pipeline([
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder(handle_unknown='ignore'))
                    
                ])
            
            preprocessor = ColumnTransformer(transformers=[
                ('num',numeric_transformer,num_columns),
                ('cat',categorical_transformer,cat_columns)
                
            ])
            
            return preprocessor
            
            
            
        except Exception as e:
            raise CustomerChurnException()
            
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method")
            
        try:
            logging.info("Starting data Transformation")
            print(self.data_validation_artifact.valid_train_file_path)
            #Load the files     
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            #seperate target 
            #training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
           
            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            #seperate numurical columns and categorical columns fron train_df and convert to list
            num_columns = input_feature_train_df.select_dtypes(include=['number']).columns.tolist()
            cat_columns  = input_feature_train_df.select_dtypes(include=['object','category']).columns.tolist()
            
            preprocessor = self.get_data_transformer_object(num_columns=num_columns,cat_columns=cat_columns)
           
           
           #####MY CODE GOES HERE
           ##saving non fitted preprocessor object
           
            save_object(self.data_transformation_config.transformed_object_file_path,preprocessor)
           
           ###END
           
            #preparing artifact
            
            data_trasformation_artifact = DataTransformationArtifact(
                transformation_object_file_path=self.data_transformation_config.transformed_object_file_path,
               
            )
            
            return data_trasformation_artifact
            
            
                
        except Exception as e:
            raise CustomerChurnException(e,sys)