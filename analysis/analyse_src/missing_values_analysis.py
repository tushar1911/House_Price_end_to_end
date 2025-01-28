from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df:pd.DataFrame):
        """
        Performs a complete missing values analysis by indetifying and visualizing missing values.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        
        Returns:
        None: This method performs the analysis and visualizes missing values.
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
    
    @abstractmethod
    def identify_missing_values(self, df:pd.DataFrame):
        """
        Indentifies missing values in the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        
        Returns:
        None: This method should print the count of missing values for each column.
        """
        pass
    
    @abstractmethod
    def visualize_missing_values(self,df:pd.DataFrame):
        """
        Visualizes missing values in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be visualized.
        
        Returns:
        None: This method should create a visualization of missing values.
        """
        pass
    
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame):
        """
        Prints the count of missing values for each columns in the dataframe.

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.
        
        Returns:
        None: Print the missing values to the console.
        """
        print("\nMissing Values count by Columns:")
        missing_values=df.isnull().sum()
        print(missing_values[missing_values>0])
        
    def visualize_missing_values(self, df):
        """
        Creates a heatmap to visualize the missing values in the dataframe.

        Parameters:
        df (_type_):The dataframe to be visualized.
        
        Returns:
        None: Displays a heatmap of missing values
        """
        print("\nVisualizing Missing Values....")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(), cbar=False)
        plt.title("Missing Values Heatmap")
        plt.show()
        
if __name__=="__main__":
    # df=pd.read_csv("extracted_data\AmesHousing.csv")
    # missing_values_analyzer=SimpleMissingValuesAnalysis()
    # missing_values_analyzer.analyze(df)
    pass
    
        