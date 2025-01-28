from abc import ABC,abstractmethod
import logging
from typing import Any

import pandas as pd 
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self, X_train:pd.DataFrame, y_train:pd.Series)-> RegressorMixin:
        """
        Abstract method to build and train a model.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        pass
    
class LinearRegressionStrategy(ModelBuildingStrategy):
    def build_and_train_model(self, X_train:pd.DataFrame, y_train:pd.Series)->Pipeline:
        """
        Builds and trains a linear regression model using scikit-learn.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        Pipeline: A scikit-learn pipeline with a trained Linear Regression model.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame")
        
        if not isinstance(y_train, pd.Series):
            raise TypeError("y_train must be a pandas series.")
        
        logging.info("Initializing Linear Regression model with scaling.")
        
        pipeline=Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )        
        
        logging.info("Training Linear Regression Model.")
        pipeline.fit(X_train,y_train)
        
        logging.info("Model training completed.")
        return pipeline
    
class ModelBuilder:
    def __init__(self, strategy:ModelBuildingStrategy):
        """
        Initializes the ModelBuilder with a specific model building strategy.

        Parameters:
        strategy (ModelBuildingStrategy): The strategy to be used for model building.
        """
        self._strategy=strategy
        
    def set_startegy(self, strategy:ModelBuildingStrategy):
        """
        Sets a new strategy for the ModelBuilder.

        Parameters:
        strategy (ModelBuildingStrategy): The new strategy to be used for model building.
        """
        logging.info("Switching model building startegy.")
        self._strategy=strategy
        
    def build_model(self, X_train:pd.DataFrame, y_train:pd.Series)->RegressorMixin:
        """
        Executes the model building and training using the current strategy.

        Parameters:
        X_train (pd.DataFrame): The training data features.
        y_train (pd.Series): The training data labels/target.

        Returns:
        RegressorMixin: A trained scikit-learn model instance.
        """
        logging.info("Building and training the model using the selected startegy.")
        return self._strategy.build_and_train_model(X_train, y_train)
    
if __name__=="__main__":
    # df=pd.read_csv("extracted_data\AmesHousing.csv")
    # X_train=df.drop(columns=['SalePrice'])
    # y_train=df['SalePrice']
    
    # model_builder=ModelBuilder(LinearRegressionStrategy())
    # trained_model=model_builder.build_model(X_train,y_train)
    pass
    