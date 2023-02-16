from src.Insurance.entity import ModelTrainingConfig
from src.Insurance import logger
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import pickle


stage = 'Model training'
class ModelTraining:
    def __init__(self, config=ModelTrainingConfig):
        self.config = config
        logger.info(f'*****************"stage {stage} started"********************')

    def train_model(self):
        train_file_path = self.config.transformed_train_dir
        train_file = os.path.join(train_file_path,'Insurance_data.csv')
        logger.info(f'loading the transformed train data for training from path {train_file_path}')
        df = pd.read_csv(train_file, index_col=[0])
        logger.info(f'shape of the transformed train data {df.shape}')
        logger.info(f'splitting the train data into independent and dependent feature ')
        X= df.iloc[:,:-1].values
        y =df.iloc[:,-1].values

        test_file_path = self.config.tranformed_test_dir
        test_file = os.path.join(test_file_path,'Insurance_data.csv')
        logger.info(f'loading the transformed test data for testing from path  {test_file_path}')
        df1 = pd.read_csv(test_file, index_col=[0])
        logger.info(f'shape of the transformed test data {df1.shape}')
        logger.info(f'splitting the test data into independent and dependent feature ')
        X1 = df1.iloc[:,:-1].values
        y1 = df1.iloc[:,-1].values

        # Define a dictionary of model names and their corresponding classes
        models = {
            'Linear Regression': LinearRegression,
            'Decision Tree': DecisionTreeRegressor,
            'Random Forest': RandomForestRegressor,
            'MLP': MLPRegressor,
            'SVM': SVR,
            'Gradient Boosting': GradientBoostingRegressor,
            'K-Nearest Neighbors': KNeighborsRegressor
        }
        logger.info(f'Loading different Ml algorithm {models}')

        # Train and evaluate all models, and store their results in a dictionary
        results = {}
        for name, model_class in models.items():
            logger.info('model is passed with train independent and dependent feature')
            model = model_class().fit(X, y)
            y_pred = model.predict(X1)
            logger.info('Evaluation of the model')
            r2 = r2_score(y1, y_pred)
            n = X1.shape[0]
            p = X1.shape[1]
            logger.info(f'model r2 score is {r2}')
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            logger.info(f'model adj_r2 score is {adj_r2}')
            mae = mean_absolute_error(y1, y_pred)
            mse = mean_squared_error(y1, y_pred)
            rmse = np.sqrt(mse)
            logger.info(f'model mae score is {mae}')
            logger.info(f'model mse score is {mse}')
            logger.info(f'model rmse score is {rmse}')
            logger.info(f'logging the model score {model}')
            results[name] = {'model': model, 'R-squared': r2, 'Adjusted R-squared': adj_r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}
            logger.info(f'Evalution of the model with different parameter {results}')
            df2=pd.DataFrame(results)
            parameter_file_path = self.config.parameter_dir
            parameter_file = os.path.join(parameter_file_path,'param.csv')
            df2.to_csv(parameter_file)
            logger.info(f'saving the evalution of all the model into csv format at {parameter_file_path}')
             # Find the best model based on your chosen evaluation metric
            logger.info('Selecting the best model amongst the different Ml algorithm started')
            best_model_name = max(results,key=lambda x: results[x]['Adjusted R-squared'])
            logger.info(f'The best Algorithmm  would be "{best_model_name}"')
            print(best_model_name)
            
            #model pickle directory location
            pickle_dir = self.config.pickle_dir
            model_location = os.path.join(pickle_dir,'best_model')

            # Store the best model for later prediction
            best_model = results[best_model_name]['model']
            with open(model_location,'wb') as f:
                pickle.dump(best_model,f)
                logger.info(f'storing the {best_model} pickle at {pickle_dir}')
            
            os.makedirs('prediction',exist_ok=True)
            path = os.path.join('prediction','best_model')
            with open(path,'wb') as f:
                pickle.dump(best_model,f)
            logger.info(f'************"stage {stage} completed"*******************')