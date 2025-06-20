import sys,os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler,RobustScaler,MinMaxScaler

from customerchurn.constant.training_pipeline import TARGET_COLUMN
from customerchurn.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from customerchurn.entity.artifact_entity import (DataTransformationArtifact,DataValidationArtifact)
from customerchurn.entity.config_entity import DataTransformationConfig
from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging
from utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_transformation_config:DataTransformationConfig,
                 data_validation_artifact:DataValidationArtifact
                 ):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact,
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
            
        except Exception as e:
            raise CustomerChurnException(e,sys)
    
    def get_data_transformer_object(cls)->Pipeline:
        '''
        this fuction will preform following preprocessing steps 
        01. handling missing values - use SimpleImputer with strategry - most_frequenct
        02. Encording - labelencoder 
        03. Scaling - Use standardScaler,robustscaler and minmax 
        
        
        '''
        logging.info("Entered get_data_transformation_object method of Transformation class")
        
        try:
            
            
            
            
            
        except Exception as e:
            raise CustomerChurnException()
            
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method")
            
        try:
            logging.info("Starting data Transformation")
            
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
            
            preprocessor = self.get_data_transformer_object()
            
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_features =  preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_features =  preprocessor_object.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_features,np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_features,np.array(target_feature_test_df)]
            
            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path,array=test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_object_file_path,preprocessor_object)
             

            save_object("final_model/preporcessor.pkl",preprocessor_object)
            
            #preparing artifact
            
            data_trasformation_artifact = DataTransformationArtifact(
                transformation_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformation_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformation_test_file_path=self.data_transformation_config.transformed_test_file_path
                
            )
            
            return data_trasformation_artifact
            
            
                
        except Exception as e:
            raise CustomerChurnException(e,sys)