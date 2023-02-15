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

class ModelTraining:
    def __init__(self, config=ModelTrainingConfig):
        self.config = config

    def train_model(self):
        train_file_path = self.config.transformed_train_dir
        train_file = os.path.join(train_file_path,'Insurance_data.csv')
        df = pd.read_csv(train_file, index_col=[0])
        X= df.iloc[:,:-1].values
        y =df.iloc[:,-1].values

        test_file_path = self.config.tranformed_test_dir
        test_file = os.path.join(test_file_path,'Insurance_data.csv')
        df1 = pd.read_csv(test_file, index_col=[0])
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

        # Train and evaluate all models, and store their results in a dictionary
        results = {}
        for name, model_class in models.items():
            model = model_class().fit(X, y)
            y_pred = model.predict(X1)
            r2 = r2_score(y1, y_pred)
            n = X1.shape[0]
            p = X1.shape[1]
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            mae = mean_absolute_error(y1, y_pred)
            mse = mean_squared_error(y1, y_pred)
            rmse = np.sqrt(mse)
            results[name] = {'model': model, 'R-squared': r2, 'Adjusted R-squared': adj_r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse}
            df2=pd.DataFrame(results)
            parameter_file_path = self.config.parameter_dir
            parameter_file = os.path.join(parameter_file_path,'param.csv')
            df2.to_csv(parameter_file)
             # Find the best model based on your chosen evaluation metric
            best_model_name = max(results,key=lambda x: results[x]['Adjusted R-squared'])
            print(best_model_name)
            
            #model pickle directory location
            pickle_dir = self.config.pickle_dir
            model_location = os.path.join(pickle_dir,'best_model')

            # Store the best model for later prediction
            best_model = results[best_model_name]['model']
            with open(model_location,'wb') as f:
                pickle.dump(best_model,f)
            # y_pred = best_model.predict(X1)
            # print(y_pred,y1)
