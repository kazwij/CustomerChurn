import yaml
from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle
from sklearn.pipeline import Pipeline


from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
            
    except Exception as e:
        raise CustomerChurnException(e,sys)
            
def write_yaml_file(file_path:str,content:object,replace:bool = False)->None:
    try:
        if replace:
             if os.path.exists(file_path):
                 os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open (file_path,"w") as file:
            yaml.dump(content,file)


    except Exception as e:
        raise CustomerChurnException(e,sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomerChurnException(e, sys) from e

def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomerChurnException(e, sys) from e

def load_numpy_array_data(file_path:str):
    try:
        with open (file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomerChurnException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models:dict,param:dict,preprocessor)->dict:
    try:
        report = {}
        best_model = None
        best_score = 0.0

        model_names = list(models.keys())

        for i in range(len(model_names)):
            model_name = model_names[i]
            model = models[model_name]
            para = param.get(model_name, {})  
            logging.info(f"model running is {model_name}")
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # Prefix model hyperparameters with "model__"
            grid_params = {"model__" + k: v for k, v in para.items()}

            gs = GridSearchCV(pipeline, grid_params, cv=3)
            gs.fit(X_train, y_train)

            best_pipeline = gs.best_estimator_

            y_train_pred = best_pipeline.predict(X_train)
            y_test_pred = best_pipeline.predict(X_test)

            train_score = recall_score(y_train, y_train_pred)
            test_score = recall_score(y_test, y_test_pred)

            report[model_name] = test_score

            if test_score > best_score:
                best_score = test_score
                best_model = best_pipeline
                best_model_name = model_name
                logging.info(f"Best model: {best_model_name} with Test Recall: {best_score:.4f}")
        
        logging.info(f"Best model selected: {best_model_name} with final Test Recall: {best_score:.4f}")

        return report, best_model

    except Exception as e:
        raise CustomerChurnException(e, sys)