import logging
from abc import ABC,abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Abstract method to apply feature engineering transformation to the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: A dataframe with the applied transformations.
        """
        pass
    
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform.

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features=features
        
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Applies a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        logging.info(f"Applying log transformation to features:{self.features}")
        df_transformed=df.copy()
        for feature in self.features:
            df_transformed[feature]=np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed
    
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (list): The list of features to apply the standard scaling to.
        """
        self.features=features
        self.scaler=StandardScaler()
        
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        logging.info(f"Applying standard scaling to features: {self.features}")
        df_transformed=df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df[self.features])
        logging.info("Standard scaling completed.")
        return df_transformed
    
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0,1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max scaling to.
        feature_range (tuple): The target range for scaling, default is (0, 1).
        """
        self.features=features
        self.scaler=MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Applies Min-Max scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """     
        logging.info(f"Applying Mix-Max scaling to features: {self.features} with range {self.scaler.feature_range}")
        df_transformed=df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df[self.features])
        logging.info("Min-Max scaling completed")
        return df_transformed
    
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features=features
        self.encoder=OneHotEncoder(sparse=False, drop="first")
        
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Applying one-hot encoding to features: {self.features}")
        df_transformed=df.copy()
        encoded_df=pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed=df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed=pd.concat([df_transformed,encoded_df], axis=1)
        logging.info("One-hot encoding completed")
        return df_transformed
    
class FeatureEngineer:
    def __init__(self, strategy:FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEngineeringStrategy): The strategy to be used for feature engineering.
        """
        self._strategy=strategy
        
    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Switching feature engineering strategy")
        self._strategy=strategy
        
    def apply_feature_engineering(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Executes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature engineering startegy.")
        return self._strategy.apply_transformation(df)
    
if __name__=="__main__":
    # df=pd.read_csv("extracted_data\AmesHousing.csv")
    # log_transformer=FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed=log_transformer.apply_feature_engineering(df)
    
    # standard_scaler=FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled=standard_scaler.apply_feature_engineering(df)
    
    # minmax_scaler=FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_minmax_scaler=minmax_scaler.apply_feature_engineering(df)
    
    # onehot_encoder=FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded=onehot_encoder.apply_feature_engineering(df)
    pass
    
    
        
                
    
