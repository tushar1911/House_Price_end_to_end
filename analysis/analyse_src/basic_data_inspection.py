from abc import ABC,abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df:pd.DataFrame):
        """
        Perform a specific type of data inspection.
        
        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.
        
        Returns:
        None: This method prints the inspection results directly.
        """
        pass

class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data type and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame):The dataframe to be inspected
        
        Returns:
        None:Print the data type and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())
        
    
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected 
        
        Returns:
        None: Prints summary statistics to the console.
        """
        print("\nSummary Statistics (Numerical Features)")
        print(df.describe())
        print("\nSummary Statistics (Categorical Features)")
        print(df.describe(include=['object']))
        
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy. 

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.
        
        Returns:
        None
        """
        self._strategy=strategy
        
    def set_strategy(self, strategy:DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector

        Paramters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.
        
        Returns:
        None
        """
        self._strategy=strategy
    
    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection with the current strategy.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be inspected
        
        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)
        
if __name__=="__main__":
    
    # df=pd.read_csv("extracted_data\AmesHousing.csv")
    # inspector=DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)
    
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    pass
                   
    
        