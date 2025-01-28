import logging
from typing import Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_evaluator import ModelEvaluator, RegressionModelEvaluationStrategy
from zenml import step

@step(enable_cache=False)
def model_evaluator_step(
    trained_model:Pipeline, X_test:pd.DataFrame, y_test: pd.Series
)-> Tuple[dict,float]:
    """
    Evaluates the trained model using ModelEvaluator and RegressionModelEvaluationStrategy.

    Parameters:
    trained_model (Pipeline): The trained pipeline containing the model and preprocessing steps.
    X_test (pd.DataFrame): The test data features.
    y_test (pd.Series): The test data labels/target.

    Returns:
    dict: A dictionary containing evaluation metrics.
    """
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas Dataframe")
    if not isinstance(y_test, pd.Series):
        raise TypeError("y_test must be a pandas series.")
    
    logging.info("Applying the same preprocessing to the test data.")
    
    X_test_processed=trained_model.named_steps['preprocessor'].transform(X_test)
    evaluator=ModelEvaluator(strategy=RegressionModelEvaluationStrategy())
    evaluation_metrics= evaluator.evaluate(
        trained_model.named_steps['model'], X_test_processed, y_test
    )
    if not isinstance(evaluation_metrics,dict):
        raise ValueError("Evaluation metrics must be returned as a dictionary.")
    mse=evaluation_metrics.get("Mean Squared Error", None)
    return evaluation_metrics, mse