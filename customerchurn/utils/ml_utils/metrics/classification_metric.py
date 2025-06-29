from customerchurn.entity.artifact_entity import ClassifactionMetricArtifact
from customerchurn.exceptions.exception import CustomerChurnException
from sklearn.metrics import f1_score,precision_score,recall_score
import sys

def get_classification_score(y_true,y_pred)->ClassifactionMetricArtifact:
    try:
            
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precission_score=precision_score(y_true,y_pred)

        classification_metric =  ClassifactionMetricArtifact(f1_score=model_f1_score,
                    precission_score=model_precission_score, 
                    recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise CustomerChurnException(e,sys)