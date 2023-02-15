from src.Insurance import logger
from src.Insurance.config import Configuration
from src.Insurance.components import DataIngestion ,DataTransformation,ModelTraining
import pymongo
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import os

MONGODB_URL = os.getenv('MONGODB_URL')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

def main():
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    # data_ingestion.download_data()
    # data_ingestion.split_data()
    data_ingestion.Mongo_data()


def main1():
    config = Configuration()
    data_transform_config = config.get_data_transformation_config()
    data_transform = DataTransformation(data_transform_config)
    data_transform.get_train_data()
    data_transform.get_test_data()


def main2():
    config= Configuration()
    model_training_config = config.model_training_config()
    model_training = ModelTraining(model_training_config)
    model_training.train_model()


if __name__ =='__main__':
    try:
        main()
        main1()
        main2()
    except Exception as e:
        logger.exception(e)