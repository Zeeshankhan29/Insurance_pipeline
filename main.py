from src.Insurance import logger
from src.Insurance.config import Configuration
from src.Insurance.components import DataIngestion ,DataTransformation,ModelTraining,ModelPusher
import pymongo
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import os

MONGODB_URL = os.getenv('MONGODB_URL')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

def main1():
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    # data_ingestion.download_data()
    # data_ingestion.split_data()
    data_ingestion.Mongo_data()


def main2():
    config = Configuration()
    data_transform_config = config.get_data_transformation_config()
    data_transform = DataTransformation(data_transform_config)
    data_transform.get_train_data()
    data_transform.get_test_data()



# apt-get update && apt install awscli -y

def main3():
    config= Configuration()
    model_training_config = config.model_training_config()
    model_training = ModelTraining(model_training_config)
    model_training.train_model()


def main4():
    model_pusher_config = Configuration()
    model_pusher = model_pusher_config.model_pusher_config()
    model_pusher  = ModelPusher(model_pusher)
    model_pusher.model_pusher_s3_aws()
    model_pusher.get_s3_bucket_model()

if __name__ =='__main__':
    try:
        main1()
        main2()
        main3()
        main4()
    except Exception as e:
        logger.exception(e)