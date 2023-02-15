from src.Insurance.config import Configuration
from src.Insurance.components import DataTransformation
from src.Insurance import logger

def main1():
    config = Configuration()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(data_transformation_config)
    data_transformation.get_train_data()
    data_transformation.get_test_data()



if __name__=='__main__':
    try:
        main1()
    except Exception as e:
        logger.exception(e)