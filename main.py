from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging

from customerchurn.components.data_ingestion import DataIngestion
from customerchurn.components.data_validation import DataValidation
from customerchurn.components.data_trasnformation import DataTransformation

from customerchurn.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from customerchurn.entity.config_entity import TrainingPipelineConfig

from customerchurn.components.model_training import ModelTrainer
from customerchurn.entity.config_entity import ModelTrainerConfig
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
        
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("Data transformation start")
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        print(data_transformation_artifact)
        logging.info("Data transformation completed")
        
        logging.info("Model Training Started")
        model_training_config = ModelTrainerConfig(trainingpipelineconfig)
        logging.info("Model training config completed")
        model_trainer = ModelTrainer(model_trainer_config=model_training_config, data_transformation_artifact = data_transformation_artifact,data_validation_artifact=data_validation_artifact )
        model_trainer_artifact  = model_trainer.initiat_model_trainer()
        
        logging.info("Model Training artifact created")
             
    except Exception as e:
        raise CustomerChurnException(e,sys)