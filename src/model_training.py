from math import log
import os
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.data_ingestion import load_params

log_dir='logs'

logger=logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} with shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse csv file: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found :{e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading data: {e}")
        raise
    
def train_model(x_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    try:
        if x_train.shape[0]!=y_train.shape[0]:
            raise ValueError("NUmber of samples in x_train and y_train do not match")
        
        logger.debug("Intiallizing RandomForest model with parameters: %s",params)
        model=RandomForestClassifier(**params)
        
        logger.debug("Model training started with %d samples",x_train.shape[0])
        model.fit(x_train,y_train)
        logger.debug("Model Training Completed Successfully")
        return model
    
    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise

def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug(f"Model saved to {file_path}")
    
    except FileNotFoundError as e:
        logger.error(f"Directory not found while saving the model: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while saving the model: {e}")
        raise
        

def main():
    try:
        params=load_params('params.yaml')['model_training']
        train_data=load_data('src/data/processed/train_tfidf.csv')
        x_train = train_data.iloc[:, :-1].values
        y_train=train_data.iloc[:,-1].values
        
        clf=train_model(x_train,y_train,params)
        model_save_path='src/models/model.pkl'
        save_model(clf,model_save_path)
    
    except Exception as e:
        logger.error(f"Failled to complete the model training process: {e}")
        print(f"Error: {e}")

if __name__=='__main__':
    main()