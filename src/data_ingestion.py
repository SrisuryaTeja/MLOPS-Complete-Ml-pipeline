import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split





logger=logging.getLogger('data_ingestion')

log_dir='logs'

os.makedirs(log_dir,exist_ok=True)

logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def laod_data(data_url:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"unexpected error occurred while loading data:{e}")
        raise 

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug("Data preprocessing completed successfully")
        return df
    except KeyError as e :
        logger.error(f"Column not found during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error(f"Failed to create raw data directory: {e}")
        raise


def main():
    data_path="https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv"
    try:
        df=laod_data(data_path)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=0.2,random_state=2)
        save_data(train_data,test_data,'src/data')
    except Exception as e:
        logger.error(f"Fail to complete the data ingestion process:{e}") 
        raise


if  __name__=="__main__":
    main()