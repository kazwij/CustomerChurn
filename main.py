from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging

from customerchurn.components.data_ingestion import DataIngestion
from customerchurn.components.data_validation import DataValidation
#from customerchurn.components.data_transformation import DataTransformation

from customerchurn.entity.config_entity import DataIngestionConfig,DataValidationConfig
from customerchurn.entity.config_entity import TrainingPipelineConfig

#from customerchurn.components.model_trainer import ModelTrainer
#from customerchurn.entity.config_entity import ModelTrainerConfig 
import sys


if __name__ =='__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)

        datavalidationconfig = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(dataingestionartifact,datavalidationconfig)
        logging.info("Initiate the Data Validation")
        data_validation_artifact = data_validation.initial_data_validation()
        logging.info("data validation completed")
        print(data_validation_artifact) 
               
    except Exception as e:
        raise CustomerChurnException(e,sys)