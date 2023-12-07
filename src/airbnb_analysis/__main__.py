# __main__.py within the airbnb_analysis package

import matplotlib.pyplot as plt
from airbnb_analysis.summary import DataSummary
from airbnb_analysis.analysis import ExploratoryDataAnalysis
from airbnb_analysis.inference import Inference
from airbnb_analysis.modeling import Modeling

def main():
    """
    Run Airbnb data analysis as a script, processing data through various stages of analysis.
    """
    print("------------------------------------------------")
    print("Airbnb Data Analysis")
    print("------------------------------------------------")
    
    # Load data
    data_path = "data/listings.csv" 
    data = pd.read_csv(data_path)
    
    # Data Summary
    data_summary = DataSummary(data)
    data_summary.data_info()
    data_summary.missing_value_summary()

    # Exploratory Data Analysis
    eda = ExploratoryDataAnalysis(data)
    eda.plot_price_distribution()
    eda.plot_price_distribution_seaborn()
    eda.plot_minimum_nights_distribution()
    eda.plot_minimum_nights_distribution_seaborn()

    # Inference
    inference = Inference(data)
    price_room_type_test = inference.hypothesis_test_price_room_type()
    print(f"ANOVA test for price across room types: {price_room_type_test}")

    # Predictive Modeling
    target = 'price'  # Update with actual target variable
    modeling = Modeling(data, target)
    modeling.train()
    modeling_results = modeling.evaluate()
    print(f"Modeling results: {modeling_results}")
    
    # Visualizations
    modeling.plot_residuals()
    modeling.plot_actual_vs_predicted()
    
    # Show any plots that were created
    plt.show()

