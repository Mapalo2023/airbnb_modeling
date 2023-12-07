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
    summarize_data()
    explore_data()
    conduct_inference()
    build_and_evaluate_model()
    print("------------------------------------------------")
    
def summarize_data():
    """Generate summary statistics of the Airbnb dataset."""
    summary = DataSummary('data/listings.csv')
    summary.generate_summary_statistics()

def explore_data():
    """Perform exploratory data analysis on the Airbnb dataset."""
    eda = ExploratoryDataAnalysis('data/listings.csv')
    eda.perform_exploratory_analysis()

def conduct_inference():
    """Conduct inferential statistics to test hypotheses related to the Airbnb dataset."""
    inference = Inference('data/listings.csv')
    inference.perform_hypothesis_testing()

def build_and_evaluate_model():
    """Build a predictive model and evaluate its performance."""
    modeling = Modeling('data/listings.csv', target='price')
    modeling.train_model()
    modeling.evaluate_model()
    
    # Visualize model results
    modeling.plot_model_results()
    plt.show()

if __name__ == "__main__":
    main()
