import pymongo
import json
import pandas as pd
from src.Insurance import logger
from dotenv import load_dotenv
load_dotenv()
import os
MONGODB_URL = os.getenv('MONGODB_URL')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')




client = pymongo.MongoClient(MONGODB_URL)
DATA_FILE_PATH ='C:/Zeeshan/Vscode_projects/Insurance_pipeline/Insurance.csv'


if __name__ == '__main__':
    try:
        logger.info('**********started****************')
        logger.info('Data loading has started')
        df = pd.read_csv(DATA_FILE_PATH)
        column = df.columns[0]
        df = df.drop(columns=column)
        logger.info(f'shape of dataframe is {df.shape}')
        json_data = df.to_json(orient='records')
        logger.info('Data is converted to json format')
        data = json.loads(json_data)
        DATABASE = client[DATABASE_NAME]
        COLLECTION = DATABASE[COLLECTION_NAME]
        COLLECTION.insert_many(data)
        logger.info('Data is dumped into Mongodb')
        logger.info('**************completed***************')
    
    except Exception as e:
        logger.exception(str(e))
        print('Something went wrong Please check log')

