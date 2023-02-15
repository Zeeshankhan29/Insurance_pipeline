from src.Insurance.entity.config_entity import DataTransformationConfig
from src.Insurance import logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
import pandas as pd
import os
import pickle


stage = 'DataTransformation'


class DataTransformation:
    def __init__(self,config=DataTransformationConfig):
        self.config=config
        logger.info(f'*******************"stage {stage} started"*******************')

    @staticmethod    
    def _outliers(df):
        for i in df.columns:
            q1=df.loc[:,[i]].quantile(0.25)
            q3=df.loc[:,[i]].quantile(0.75)
            IQR = q3-q1
            lower = q1 - (1.5*IQR)
            upper = q3 + (1.5*IQR)
            df.loc[df[i]<=lower[0],i]=lower[0]
            df.loc[df[i]>=upper[0],i]=upper[0]
        return df

    def get_train_data(self):
        path = self.config.train_dir
        filepath = os.path.join(path,'Insurance_data.csv')
        df =pd.read_csv(filepath,index_col=[0])
        y=df.iloc[:,-1]
        df.drop_duplicates(inplace=True)
        logger.info(f'Reading train data has started from {filepath}')
        logger.info(f'Shape of the Dataframe is {df.shape}')
        cat_feat =df.dtypes[df.dtypes!='O'].index
        cat_feat=cat_feat[:-1]
        logger.info('filtering numerical feature to check for outliers')
        
        df[cat_feat] = self._outliers(df[cat_feat])
        logger.info(f'Outliers handling has completed for numerical features {cat_feat}')
        
        # Identify the columns that should be scaled
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scale_cols = num_cols.drop(df.columns[-1])

        # Identify the columns that should be encoded
        encode_cols = df.select_dtypes(include=['object']).columns

        # Create the transformers: a StandardScaler for the scaled columns and a OneHotEncoder for the encoded columns
        scale_transformer = StandardScaler()
        encode_transformer = OrdinalEncoder()

        # Use the ColumnTransformer to apply the transformers to the appropriate columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', scale_transformer, scale_cols),
                ('encode', encode_transformer, encode_cols)
            ])
        logger.info(f'Applying Scaling to numerical features {scale_cols}')
        logger.info(f'Applying Onehotencoding to categorical features {encode_cols}')

        # Use a pipeline to apply the ColumnTransformer and any other desired processing steps
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
        ])

        # Fit the pipeline to the data
        pipeline.fit(df)
        filepath = self.config.pickle_dir
        filename='pipeline_object'
        pickle_file_path = os.path.join(filepath,filename)
        with open(pickle_file_path,'wb') as f:
            pickle.dump(pipeline,f) 

        # Transform the data
        transformed_data = pipeline.transform(df)

        num_columns= list(scale_cols)
        cat_columns= list(encode_cols)
        feature_names = num_columns+cat_columns

        
        df=pd.DataFrame(transformed_data,columns=feature_names)
        logger.info(f'Shape of the dataframe after transform is {df.shape}')
        transform_filepath = self.config.transformed_train_dir
        transform_filename = os.path.join(transform_filepath,'Insurance_data.csv')
        y = y.reset_index()
        df = pd.concat([df,y],axis=1)
        df = df.drop(columns=['index'])
        df.to_csv(transform_filename)
        logger.info(f'Saving the transformed train data at {transform_filepath} with filename Insurance_data.csv')
       
        
    def get_test_data(self):
        path1 = self.config.test_dir
        filepath = os.path.join(path1,'Insurance_data.csv')
        df =pd.read_csv(filepath,index_col=[0])
        y=df.iloc[:,-1]
        df.drop_duplicates(inplace=True)
        logger.info(f'Reading test data has started from {filepath}')
        logger.info(f'Shape of the Dataframe is {df.shape}')
        cat_feat =df.dtypes[df.dtypes!='O'].index
        cat_feat=cat_feat[:-1]
        logger.info('filtering numerical feature to check for outliers')
        
        df[cat_feat] = self._outliers(df[cat_feat])
        logger.info(f'Outliers handling has completed for numerical features {cat_feat}')
        
        # Identify the columns that should be scaled
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scale_cols = num_cols.drop(df.columns[-1])

        # Identify the columns that should be encoded
        encode_cols = df.select_dtypes(include=['object']).columns

        # Create the transformers: a StandardScaler for the scaled columns and a OneHotEncoder for the encoded columns
        scale_transformer = StandardScaler()
        encode_transformer = OrdinalEncoder()

        # Use the ColumnTransformer to apply the transformers to the appropriate columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('scale', scale_transformer, scale_cols),
                ('encode', encode_transformer, encode_cols)
            ])
        logger.info(f'Applying Scaling to numerical features {scale_cols}')
        logger.info(f'Applying Onehotencoding to categorical features {encode_cols}')

        # Use a pipeline to apply the ColumnTransformer and any other desired processing steps
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
        ])

        # Fit the pipeline to the data
        pipeline.fit(df)
        
        # Transform the data
        transformed_data = pipeline.transform(df)

        num_columns= list(scale_cols)
        cat_columns= list(encode_cols)
        feature_names = num_columns+cat_columns

        
        df=pd.DataFrame(transformed_data,columns=feature_names)
        logger.info(f'Shape of the dataframe after transform is {df.shape}')
        transform_filepath = self.config.transformed_test_dir
        transform_filename = os.path.join(transform_filepath,'Insurance_data.csv')
        y = y.reset_index()
        df = pd.concat([df,y],axis=1)
        df = df.drop(columns=['index'])
        df.to_csv(transform_filename)
        logger.info(f'Saving the transformed test data at {transform_filepath} with filename Insurance_data.csv')
        logger.info(f'*******************"stage {stage} completed"*******************')
    


  