import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

class Inference:
    """
    The Inference class is used for conducting various inference-based analyses 
    on a given dataset. It provides functionalities for hypothesis testin and generating statistical summaries.
    
    Attributes:
        data (pd.DataFrame): A DataFrame containing the dataset for analysis.
    """
    
    def __init__(self, data):
        """
        Initializes the Inference class with a dataset.

        Args:
            data (pd.DataFrame): The dataset to be used for inference.
        """
        self.data = data

    def hypothesis_test_price_room_type(self):
        """
        Conducts an ANOVA test to determine if there are statistically 
        significant differences in prices across different room types.

        This method assumes that the dataset has 'room_type' and 'price' columns.

        Returns:
          dict: A dictionary containing the F-statistic and the p-value of the test.
        """
         # Check if required columns are in the dataset
        if 'room_type' not in self.data or 'price' not in self.data:
            raise ValueError("Dataset must contain 'room_type' and 'price' columns.")
 
         # Prepare data for ANOVA
        groups = self.data.groupby('room_type')['price']

         # Conduct ANOVA
        f_value, p_value = stats.f_oneway(*[group for name, group in groups])

         # Return the F-statistic and p-value
        return {'F-Statistic': f_value, 'p-value': p_value}


    def statistical_summary(self, column):
        """
        Provides a statistical summary for a specified column in the dataset.

        Args:
            column (str): The name of the column for which the summary is required.

        Returns:
            pd.Series: A series containing descriptive statistics of the column.
        """
        return self.data[column].describe()

    # Add more methods for different types of inference as needed
    
    
    def scatter_plot_price_minimum_nights(self):
        """
        Creates a scatter plot between the 'price' and 'minimum_nights' columns of the data.
        """
        self.data.plot.scatter(x='price', y='minimum_nights', title="minimum_nights")
        
        
    def scatter_plot_price_minimum_nights_seaborn(self):
        """
        Creates a scatter plot between the 'price' and 'minimum_nights' columns of the data using Seaborn.
        
        Returns:
        - None. Displays the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='price', y='minimum_nights')
        plt.title("Scatter plot of Price vs Minimum Nights")
        plt.show()

    def scatter_plot_minimum_nights_number_of_reviews(self):
        """
        Creates a scatter plot between the 'minimum_nights' and 'number_of_reviews' columns of the data.
        """
        self.data.plot.scatter(x='minimum_nights', y='number_of_reviews', title="minimum_nights")
        
    
    def scatter_plot_minimum_nights_number_of_reviews_seaborn(self):
        """
        Creates a scatter plot between the 'minimum_nights' and 'number_of_reviews' columns of the data using Seaborn.
        
        Returns:
        - None. Displays the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='minimum_nights', y='number_of_reviews')
        plt.title("Scatter plot of Minimum Nights vs Number of Reviews")
        plt.show()

        
    def catplot_price_neighbourhood(self, figsize=(10, 6)):
        """
        Creates a categorical scatter plot between the 'price' and 'neighbourhood' columns of the data.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (10, 6).
        
        Returns:
        - None. Displays the plot.
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        plt.figure(figsize=figsize)
        sns.catplot(data=self.data, x="price", y="neighbourhood", alpha=0.6, height=figsize[1], aspect=figsize[0]/figsize[1])
        plt.xscale('log')  # Set log scale on x-axis for better visualization if data is skewed
        plt.title("Categorical scatter plot of Price vs Neighbourhood")
        plt.show()

    def catplot_price_room_type(self, figsize=(10, 6)):
        """
        Creates a categorical scatter plot between the 'price' and 'room_type' columns of the data.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (10, 6).
        
        Returns:
        - None. Displays the plot.
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        plt.figure(figsize=figsize)
        sns.catplot(data=self.data, x="price", y="room_type", alpha=0.6, height=figsize[1], aspect=figsize[0]/figsize[1])
        plt.xscale('log')  # Set log scale on x-axis for better visualization if data is skewed
        plt.title("Categorical scatter plot of Price vs Room Type")
        plt.show()
        
    def crosstab_room_type_neighbourhood(self):
        """
        Computes a crosstab of counts between the 'room_type' and 'neighbourhood' columns of the data.

        Returns:
        - crosstab (DataFrame): The computed crosstab DataFrame.
        """
        crosstab = pd.crosstab(index=self.data['room_type'], columns=self.data['neighbourhood'])
        return crosstab
    
    def plot_crosstab_bar(self):
        """
        Computes a crosstab of counts between the 'room_type' and 'neighbourhood' columns of the data,
        and creates a bar plot of this crosstab.

        Returns:
        - ax (Axes): The axes object with the plot.
        """
        crosstab = pd.crosstab(index=self.data['room_type'], columns=self.data['neighbourhood'])
        ax = crosstab.plot.bar(figsize=(12, 7), rot=0)
        ax.set(ylabel='Count', title='Neighbourhood vs. Room Type')
        ax.legend(title='Neighbourhood', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        return ax
    
    def catplot_price_room_type_neighbourhood(self):
        """
        Creates a categorical bar plot displaying the average price for each room type,
        separated by neighbourhood.

        Returns:
        - g (FacetGrid): The Seaborn FacetGrid object with the plot.
        """
        g = sns.catplot(x='price', y='room_type', hue='neighbourhood', kind='bar', data=self.data, legend=False)
        g.fig.set_figwidth(12)
        g.fig.set_figheight(7)
        g.ax.legend(title='Neighbourhood', bbox_to_anchor=(1.05, 1), loc='upper left')
        g.set_axis_labels("Average Price", "Room Type")
        return g
    
    def correlation_analysis(self, columns):
        """
        Performs correlation analysis on specified columns.

        Parameters:
            columns (list): List of columns to include in correlation analysis.

        Returns:
            A DataFrame with correlation values between specified columns.
        """
        return self.data[columns].corr()

    def multi_variate_analysis(self, x, y, hue=None, kind='scatter'):
        """
        Conducts multivariate analysis using seaborn pairplot or catplot.

        Parameters:
            x (str): X-axis variable.
            y (str): Y-axis variable.
            hue (str): Variable for color encoding.
            kind (str): The kind of plot to draw
        """
        if kind == 'scatter':
            sns.scatterplot(data=self.data, x=x, y=y, hue=hue)
        elif kind == 'line':
            sns.lineplot(data=self.data, x=x, y=y, hue=hue)
        elif kind == 'bar':
            sns.barplot(data=self.data, x=x, y=y, hue=hue)
        # Add more plot types as needed
        else:
            raise ValueError(f"Plot kind '{kind}' is not supported")

        plt.title(f'{kind.capitalize()} Plot of {y} vs {x}')
        plt.show()

 


