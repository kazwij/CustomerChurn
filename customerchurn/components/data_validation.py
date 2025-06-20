
from customerchurn.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from customerchurn.entity.config_entity import DataValidationConfig
from customerchurn.constant.training_pipeline import SCHEMA_FILE_PATH
from customerchurn.utils.main_utils.utils import read_yaml_file, write_yaml_file
from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging
import os,sys
import pandas as pd
from scipy.stats import ks_2samp,chi2_contingency


class DataValidation:
    
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e,sys)

    @staticmethod    
    def read_data(file_path:str)->pd.DataFrame:
        
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        
        try:
            number_of_columns = len(self.schema_config['columns'])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has colums : {len(dataframe.columns)}")
            
            if len(dataframe.columns) == number_of_columns:
                return True
            
            return False
        
        except Exception as e:
            raise CustomerChurnException(e,sys)
            
    
    def check_for_numerical_column(self,dataframe:pd.DataFrame)->bool:
        
        try:
            
            if set(self.schema_config['numerical_columns']) == set(dataframe.select_dtypes(include='number').columns):
                return True
            return False
            
        except Exception as e:
            raise CustomerChurnException(e,sys)
        
    
    def detect_dataset_drift(self,base_df,current_df, threshold=0.05)->bool:
        '''
        use ks_2samp to check nurerical drift
        use chi2 to check categorical drift 
        update report 
        return drift status
        '''
        try:
            status = True
            report = {}
            
            for col in base_df.columns:
                d1 = base_df[col].dropna()
                d2 = current_df[col].dropna()
                
                if pd.api.types.is_numeric_dtype(d1):
                    #working on numerical columns
                    drift_numeric = ks_2samp(d1,d2)
                    p_value = drift_numeric.pvalue
                    
                else:
                    #for categorical values columns
                    d1_count = d1.value_counts(normalize=True)
                    d2_count = d2.value_counts(normalize=True)
                    
                    all_categories = set(d1_count.index).union(set(d2_count.index))
                    d1_aligned = d1_count.reindex(all_categories,fill_value=0)
                    d2_aligned = d2_count.reindex(all_categories,fill_value =0)
                    
                    contingency = pd.DataFrame([d1_aligned,d2_aligned])
                    
                    try:
                        chi2_stat,p_value,_,_ = chi2_contingency(contingency)
                    except Exception as e:
                        raise CustomerChurnException(e,sys)
                
                is_drifiting = p_value<threshold
                 
                if is_drifiting:
                    is_found = False
                
                report[col] = {
                    "p_value" : float(p_value),
                    "Is Stable" : not is_drifiting
                    
                }        
                
            #get drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            #make dir
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path = drift_report_file_path,content=report)
            
        
        except Exception as e:
            raise CustomerChurnException(e,sys)
        

        
    def initial_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data from the train and test 
            train_dataframe  = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            
            ## validate number of columns 
            status =self.validate_number_of_columns(train_dataframe)
            if not status:
                erorr_message =f"Train dataframe does not contain all columns.\n"
            
            
            status =self.validate_number_of_columns(test_dataframe)
            if not status:
                erorr_message=f"Test dataframe does not contain all colunmns. \n"

            status = self.check_for_numerical_column(train_dataframe)
            if not status:
               erorr_message=f"Train dataframe does not contain all numerical columns. \n" 

            status = self.check_for_numerical_column(test_dataframe)
            if not status:
               erorr_message=f"Test dataframe does not contain all numerical columns. \n"

            ## check data drift 
            status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path,index=False,header=False
            )

            test_dataframe.to_csv(

                self.data_validation_config.valid_test_file_path,index=False,header=False
            )

            data_validation_artifact = DataValidationArtifact(

                validation_status = status,
                valid_train_file_path = self.data_ingestion_artifact.train_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path

            )
            return data_validation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)