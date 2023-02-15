from src.Insurance.config import Configuration
from src.Insurance.components import DataIngestion
from src.Insurance import logger


def main():
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    # data_ingestion.download_data()
    # data_ingestion.split_data()
    data_ingestion.Mongo_data()

if __name__ =='__main__':
    try:
        main()
    except Exception as e:
        logger.exception(e)