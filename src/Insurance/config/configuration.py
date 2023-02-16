from src.Insurance.utils import create_directories,read_yaml
from src.Insurance.entity import DataIngestionConfig,DataTransformationConfig,ModelTrainingConfig,ModelPusherConfig
from src.Insurance.constants import CONFIG_FILE_PATH

class Configuration:
    def __init__(self,config_filepath=CONFIG_FILE_PATH):
        self.config=read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.raw_data_dir])
        create_directories([config.train_dir])
        create_directories([config.test_dir])
        create_directories([config.data_dir])



        data_ingestion = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            raw_data_dir=config.raw_data_dir,
            train_dir = config.train_dir,
            test_dir=config.test_dir,
            data_dir = config.data_dir)


        return data_ingestion


    def get_data_transformation_config(self):
        transform_config = self.config.data_transformation
        create_directories([transform_config.root_dir])
        create_directories([transform_config.transformed_train_dir])
        create_directories([transform_config.transformed_test_dir])
        create_directories([transform_config.pickle_dir])
        
        data_transform = DataTransformationConfig(
            train_dir=transform_config.train_dir,
            test_dir=transform_config.test_dir,
            root_dir=transform_config.root_dir,
            pickle_dir=transform_config.pickle_dir,
            transformed_train_dir=transform_config.transformed_train_dir,
            transformed_test_dir=transform_config.transformed_test_dir)
        
        return data_transform


    def model_training_config(self):
        model_training_config =self.config.model_training
        create_directories([model_training_config.pickle_dir])
        create_directories([model_training_config.parameter_dir])
        
        model_training = ModelTrainingConfig(
            tranformed_test_dir=model_training_config.transformed_train_dir,
            transformed_train_dir=model_training_config.transformed_test_dir,
            pickle_dir=model_training_config.pickle_dir,
            parameter_dir=model_training_config.parameter_dir)

        return model_training
    

    def model_pusher_config(self):
        model_pusher_config = self.config.model_pusher
        create_directories([model_pusher_config.s3_bucket_pickle])

        model_pusher =  ModelPusherConfig(
             
             pickle_dir=model_pusher_config.pickle_dir,
             s3_bucket_pickle=model_pusher_config.s3_bucket_pickle
                )
        
        return model_pusher