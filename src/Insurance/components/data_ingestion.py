from src.Insurance.utils.common import create_directories,get_size
from src.Insurance.entity import DataIngestionConfig
from src.Insurance import logger
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import pymongo
import urllib
from dotenv import load_dotenv
load_dotenv()



stage= 'Data Ingestion'

class DataIngestion:
    _MONGODB_URL = os.getenv('MONGODB_URL')
    _DATABASE_NAME = os.getenv('DATABASE_NAME')
    _COLLECTION_NAME = os.getenv('COLLECTION_NAME')
    
    def __init__(self,data_ingestion_config=DataIngestionConfig):
        self.data_ingestion_config=data_ingestion_config
        logger.info(f'**************"stage {stage} started "*****************')
    
      


    def download_data(self):
        github_url_data =  self.data_ingestion_config.source_URL
        path = Path(self.data_ingestion_config.data_dir)
        file_name = os.path.basename(github_url_data)
        path1 = os.path.join(path,file_name)
        if not os.path.exists(path1) or os.path.getsize(path1)==0:
            file_name = os.path.basename(github_url_data)
            data_from_url = urllib.request.urlretrieve(github_url_data, os.path.join(path, file_name))
            logger.info('Data Download Started')
            return data_from_url

        else:
            file_name = os.path.basename(github_url_data)
            data_from_url = urllib.request.urlretrieve(github_url_data, os.path.join(path, file_name))
            logger.info('Data already Exists')
            return data_from_url

        
           



    def split_data(self):
        config = self.data_ingestion_config

        data_dir = config.data_dir
        filename = os.path.basename(config.source_URL)
        file_dir = os.path.join(data_dir,filename)
        logger.info(f'Reading {filename} data')
        df = pd.read_csv(file_dir,index_col=[0])
        logger.info('Splitting of the data into train and test dataset has begun')
        train_dataset,test_dataset = train_test_split(df,random_state=42,test_size=0.3)
        test_file_path = os.path.join(self.data_ingestion_config.test_dir,filename)
        train_file_path = os.path.join(self.data_ingestion_config.train_dir,filename)
        train_dataset.to_csv(train_file_path)
        logger.info(f'Saving the train data at {train_file_path}')
        test_dataset.to_csv(test_file_path)
        logger.info(f'Saving the test data at {test_file_path}')
        logger.info(f'**************"stage {stage} completed" *****************')


    def Mongo_data(self):
        config = self.data_ingestion_config
        client=pymongo.MongoClient(DataIngestion._MONGODB_URL)
        database = client[DataIngestion._DATABASE_NAME]
        collection = database[DataIngestion._COLLECTION_NAME]
        logger.info('Reading data from Mongodb has started')
        l=[]
        for i in collection.find():
            l.append(i)
            df=pd.DataFrame(l)
        logger.info('Data from Mongodb is  converted to Dataframe')
        df = df.iloc[:,1:]
        filepath=os.path.join(config.data_dir,'Insurance_data.csv')
        df.to_csv(filepath)
        logger.info('Saving the data in the local folder') 

        data_path = os.path.join(config.data_dir,'Insurance_data.csv')
       
        logger.info('Reading the data from the local folder ')
        dataframe =pd.read_csv(data_path)
        dataframe.drop_duplicates(inplace=True)
        dataframe = dataframe.iloc[:,1:]
       
        logger.info('Splitting of the data into train and test dataset has begun')
        train_data ,test_data = train_test_split(dataframe,random_state=42,test_size=0.2)
        train_path =os.path.join(config.train_dir,'Insurance_data.csv')
        
        logger.info(f'Saving the train data at {train_path}')
        train_data.to_csv(train_path)
        test_path = os.path.join(config.test_dir,'Insurance_data.csv')
        
        logger.info(f'Saving the test data at {test_path}')
        test_data.to_csv(test_path)
        logger.info(f'**************"stage {stage} completed" *****************')
        

