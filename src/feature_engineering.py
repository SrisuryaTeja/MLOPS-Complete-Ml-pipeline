import os
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer



log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('Feature Engineering')
logger.setLevel(logging.DEBUG)

console_handler=logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler=logging.FileHandler(os.path.join(log_dir,'feature_engineering.log'))
file_handler.setLevel(logging.DEBUG)

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str)->pd.DataFrame :
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug(f"Data loaded and missing values filled with empty strings from {file_path}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse csv file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data:{e}")
        raise

def save_data(df:pd.DataFrame,file_path:str)->None:
    
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug(f"Data saved to {file_path}")

    except Exception as e:
        logger.error(f"Unexpected Error Occurred while saving the file: {e}")
        raise

def apply_tf_idf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple :
   
   try:
           
       vectorizer=TfidfVectorizer(max_features=max_features)
       x_train=train_data['text'].values
       y_train=train_data['target'].values
       x_test=test_data['text'].values
       y_test=test_data['target'].values  
       x_train_bow=vectorizer.fit_transform(x_train)
       x_test_bow=vectorizer.transform(x_test)  
       train_df=pd.DataFrame(x_train_bow.toarray())
       train_df['label']=y_train  
       test_df=pd.DataFrame(x_test_bow.toarray())
       test_df['label']=y_test  
       logger.debug("Tfidf applied and features transformed")
       return train_df,test_df
   except Exception as e:
       logger.error(f"Error during Tfidf transformation: {e}")
       raise

def main():
    
    try:
        max_features=50
        train_data=load_data('src/data/interim/train_processed.csv')
        test_data=load_data('src/data/interim/test_processed.csv')
        
        train_df,test_df=apply_tf_idf(train_data,test_data,max_features)
        
        save_data(train_df,os.path.join('src/data','processed','train_tfidf.csv'))
        save_data(test_df,os.path.join('src/data','processed','test_tfidf.csv'))
        
    except Exception as e:
        logger.error(f"Failed to complete Feature Engineerring : {e}")
        print(f"Error : {e}")
    pass

if __name__=='__main__':
    main()