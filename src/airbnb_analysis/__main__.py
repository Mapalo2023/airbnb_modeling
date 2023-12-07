# __main__.py for the airbnb_analysis package

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
    # Assuming the data is pre-loaded and cleaned in a CSV file
    data_path = 'data/airbnb_listings.csv'  # Replace with the correct path to your data file
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
