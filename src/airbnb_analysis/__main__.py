# __main__.py for the airbnb_analysis package

import os
import pandas as pd
from airbnb_analysis.summary import DataSummary
from airbnb_analysis.analysis import ExploratoryDataAnalysis
from airbnb_analysis.inference import Inference
from airbnb_analysis.modeling import Modeling

def main():
    """
    Run Airbnb data analysis as a complete script, encompassing summary statistics,
    exploratory data analysis, inference statistics, and predictive modeling.
    """
    # Define the relative path to the listings.csv file, going up two directories
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, '..', '..', 'data')
    data_path = os.path.join(data_dir, 'listings.csv')

    # Ensure the path is correct by normalizing and converting to an absolute path
    data_path = os.path.normpath(os.path.abspath(data_path))

    # Load data
    data = pd.read_csv(data_path)

    # Data summarization
    summary = DataSummary(data)
    summary.display_head()
    summary.display_tail()
    summary.data_info()
    
    # Exploratory Data Analysis
    eda = ExploratoryDataAnalysis(data)
    eda.plot_price_distribution()
    eda.plot_minimum_nights_distribution()
    eda.plot_number_of_reviews_distribution()
    eda.plot_pairplot(data, hue='room_type')  # Replace 'room_type' with actual column name if different
    eda.plot_correlation_heatmap(data)
    
    # Inference Statistics
    inference = Inference(data)
    result = inference.hypothesis_test_price_room_type()
    print("ANOVA test result for price across room types:", result)
    
    # Predictive Modeling
    target = 'price'  # Replace with the actual target column name if different
    modeling = Modeling(data, target)
    modeling.train()
    modeling_results = modeling.evaluate()
    print("Model evaluation results:", modeling_results)
    modeling.plot_residuals()
    modeling.plot_actual_vs_predicted()
    
if __name__ == "__main__":
    main()
