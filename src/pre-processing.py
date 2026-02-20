import os
import string
import logging
import nltk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ===============================
# Ensure NLTK resources exist
# ===============================
def download_nltk_resources():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]

    for path, resource in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)



download_nltk_resources()


# ===============================
# Setup Logging
# ===============================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_path = os.path.join(log_dir, 'data_preprocessing.log')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ===============================
# Global Objects (Performance Optimized)
# ===============================
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()


# ===============================
# Text Transformation Function
# ===============================
def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)


# ===============================
# Data Preprocessing Function
# ===============================
def preprocess_df(df, text_column='text'):
    try:
        logger.debug("Starting preprocessing for DataFrame")

        df = df.drop_duplicates(keep='first')
        logger.debug("Duplicates removed")

        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug("Text column transformed")

        return df

    except KeyError as e:
        logger.error("Column not found: %s", e)
        raise
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise


# ===============================
# Main Function
# ===============================
def main(text_column='text', target_column='target'):
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug("Data loaded successfully")

        # Encode target properly (Fit only on train)
        encoder = LabelEncoder()
        train_data[target_column] = encoder.fit_transform(
            train_data[target_column]
        )
        test_data[target_column] = encoder.transform(
            test_data[target_column]
        )
        logger.debug("Target column encoded")

        # Preprocess text
        train_processed_data = preprocess_df(train_data, text_column)
        test_processed_data = preprocess_df(test_data, text_column)

        # Save processed data
        data_path = os.path.join('data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(
            os.path.join(data_path, "train_processed.csv"),
            index=False
        )
        test_processed_data.to_csv(
            os.path.join(data_path, "test_processed.csv"),
            index=False
        )

        logger.debug("Processed data saved to %s", data_path)

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
    except Exception as e:
        logger.error("Failed to complete data transformation process: %s", e)
        print(f"Error: {e}")


# ===============================
# Entry Point
# ===============================
if __name__ == '__main__':
    main()
