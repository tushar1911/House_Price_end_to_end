from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1:str, feature2: str):
        """
        Performs bivariate analysis on two features of the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        
        Returns:
        None: This method visualizes the relationship betweent the twof features
        """
        pass
    
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature1:str, feature2:str):
        """
        Plots the relationship between two numerical features using a scatter plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first numerical feature/column to be analyzed.
        feature2 (str): The name of the second numerical feature/column to be analyzed.
        
        Returns:
        None: Displays a scatter plot showing the relationship between the two features.
        """
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        
        
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature1:str, feature2:str):
        
        """
        Plots the relationship between a categorical feature and a numerical feature using a box plot.

        
        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the categorical feature/column to be analyzed.
        feature2 (str): The name of the numerical feature/column to be analyzed.
        
        Returns:
        None: Displays a box plot showing the relationship between the categorical and numerical features.
        """
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()
        
    
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the BivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for bivariate analysis.
        
        Returns:
        None
        """
        self._strategy=strategy
        
    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for bivariate analysis.
        
        Returns:
        None
        """
        self._strategy=strategy
        
    def execute_analysis(self, df:pd.DataFrame, feature1:str, feature2:str):
        """
        Executes the bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.
        
        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df,feature1,feature2)
        
if __name__=="__main__":
    # df=pd.read_csv("extracted_data\AmesHousing.csv")
    # analyzer=BivariateAnalyzer(NumericalVsNumericalAnalysis())
    # analyzer.execute_analysis(df,'Gr Liv Area', 'SalePrice')
    
    # analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    # analyzer.execute_analysis(df,'Overall Qual', 'SalePrice')
    pass

        