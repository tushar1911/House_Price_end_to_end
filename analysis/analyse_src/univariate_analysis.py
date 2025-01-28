from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df:pd.DataFrame, feature:str):
        """
        Perform univariate analysis on a specific feature of the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature(str): The name of the feature/column to be analyzed.
        
        Returns:
        None: This method visualizes the distribution of the feature.
        """
        pass

class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        """
        Plots the distribution of a numerical feature using a histogram and KDE.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the numerical feature/column to be analyzed.
        
        Returns:
        None: Displays a histogram with a KDE plot.
        """
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
        
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        """
        Plots the distribution of a categorical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the categorical feature/column to be analyzed.
        
        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature, data=df,palette="muted", hue=feature, legend=False)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
        
        
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis
        
        Returns:
        None
        """
        self._strategy=strategy
        
    def set_strategy(self, strategy:UnivariateAnalysisStrategy):
        """
        Seta a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.
        
        Returns:
        None
        """
        self._strategy=strategy
        
    def execute_analysis(self, df:pd.DataFrame, feature:str):
        """
        Executes the univariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature (str): The name of the feature/column to be analyzed.
        
        Returns:
        None: Executes the strategy's analysis method and visualized the results.
        """
        self._strategy.analyze(df,feature)
        
if __name__=="__main__":
    # df=pd.read_csv("extracted_data\AmesHousing.csv")
    # analyzer=UnivariateAnalyzer(NumericalUnivariateAnalysis())
    # analyzer.execute_analysis(df,'SalePrice')
    
    # analyzer.set_strategy(CategoricalUnivariateAnalysis())
    # analyzer.execute_analysis(df, 'Neighborhood')
    pass
        
