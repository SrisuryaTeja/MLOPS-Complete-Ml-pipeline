import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

log_file_path=os.path.join(log_dir,'model_evaluation.log')

logger=logging.getLogger("Model Evaluation")
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path:str):
    try:
        with open(file_path,'rb') as f:
            model=pickle.load(f)
        logger.debug(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the model: {e}")
        raise

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except df.errors.ParseError as e:
        logger.error(f"Failed to parse csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data: {e}")
        raise
    
def evaluate_model(clf,x_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred=clf.predict(x_test)
        y_pred_proba=clf.predict_proba(x_test)[:,1]
        
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)
        
        metric_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug(f"Model evaluation completed with metrics: {metric_dict}")
        return metric_dict
    except Exception as e:
        logger.error(f"Unexpected error occurred during model evaluation: {e}")
        raise

def save_metrics(metrics:dict,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'w') as f:
            json.dump(metrics,f,indent=4)
        logger.debug(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while saving metrics: {e}")
        raise

def main():
    try:
        clf=load_model('src/models/model.pkl')
        test_data=load_data('src/data/processed/test_tfidf.csv')
        
        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values
        metrics=evaluate_model(clf,x_test,y_test)
        save_metrics(metrics,'src/reports/metrics.json')
    except Exception as e:
        logger.error(f"Failed to complete model evaluation process: {e}")
        print(f"Error: {e}")

if __name__=="__main__":
    main()