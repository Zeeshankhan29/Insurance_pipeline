from src.Insurance.config import Configuration
from src.Insurance.components import ModelPusher
from src.Insurance import logger




def main4():
    model_pusher_config = Configuration()
    model_pusher = model_pusher_config.model_pusher_config()
    model_pusher  = ModelPusher(model_pusher)
    model_pusher.model_pusher_s3_aws()
    model_pusher.get_s3_bucket_model()



if __name__=='__main__':
    try:
        main4()
    except Exception as e:
        logger.exception(e)