from src.Insurance.config import Configuration
from src.Insurance.components import ModelTraining
from src.Insurance import logger




def main2():
    config= Configuration()
    model_training_config = config.model_training_config()
    model_training = ModelTraining(model_training_config)
    model_training.train_model()



if __name__=='__main__':
    try:
        main2()
    except Exception as e:
        logger.exception(e)