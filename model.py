# Importing Essential Libraries:

import pandas as pd
import numpy as np
import joblib


class PipelineTester:
  def __init__(self, pipeline_path: str, test_data: pd.DataFrame):
    self.pipeline_path = pipeline_path
    self.test_data = test_data

  def predict(self):
    with open(self.pipeline_path, 'rb') as file:
        loaded_pipeline = joblib.load(file)

        # Get the probability scores for each class
        probability_scores = loaded_pipeline.predict_proba(self.test_data)

        # Accessing the probability for fraudulent transaction
        fraud_class_prob = probability_scores[:, 1]
        
        return fraud_class_prob
    
