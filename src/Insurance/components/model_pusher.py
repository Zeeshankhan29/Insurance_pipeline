
from src.Insurance.config import ModelPusherConfig
import os
from src.Insurance import logger


stage='Model Pusher'
class ModelPusher:
    def __init__(self,config=ModelPusherConfig):
        self.config = config
        logger.debug(f'***************"stage {stage} started"**************************')

    
    def model_pusher_s3_aws(self):
        from_pickle_dir = self.config.pickle_dir
        logger.debug(f'Loading a pickle file from directory {from_pickle_dir}')
        bucket_name = 'demo12456'
        to_aws_dir  = self.config.pickle_dir
        logger.debug(f"Pushing the pickle file to aws s3 {bucket_name} bucket")
        os.system(f'aws s3 sync {from_pickle_dir} s3://{bucket_name}/{to_aws_dir}/')
        logger.debug('pickle model is pushed to s3 bucket completed')


    def get_s3_bucket_model(self):
        to_local_dir = self.config.s3_bucket_pickle
        bucket_name = 'demo12456'
        from_aws_dir  = self.config.pickle_dir
        logger.debug(f'Loading pickle file from {from_aws_dir}  directory from {bucket_name} bucket and storing at location {to_local_dir}')
        os.system(f'aws s3 sync s3://{bucket_name}/{from_aws_dir}/ {to_local_dir}')
        logger.debug('model extraction from s3 bucket is completed')
        logger.debug(f'*******************"stage {stage} completed" ******************************')

