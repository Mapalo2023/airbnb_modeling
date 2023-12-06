import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class ExploratoryDataAnalysis:
    """
    A class for conducting various analyses on a DataFrame.
    """
    def __init__(self, data):
        """
        Initializes the DataAnalysis object with the provided data.
        
        Parameters:
            data (pd.DataFrame): The data to be analyzed.
        """
        self.data = data

    def unique_neighbourhood(self):
        """
        Retrieves the unique values in the 'neighbourhood' column of the data.
        
        Returns:
        np.ndarray: An array containing unique neighbourhoods.
        """
        return self.data.neighbourhood.unique()

    def neighbourhood_value_counts(self):
        """
        Counts the occurrences of each unique value in the 'neighbourhood' column of the data.
        
        Returns:
        pd.Series: A Series containing the counts of unique neighbourhoods.
        """
        return self.data['neighbourhood'].value_counts()

    def unique_room_type(self):
        """
        Retrieves the unique values in the 'room_type' column of the data.
        
        Returns:
        np.ndarray: An array containing unique room types.
        """
        return self.data.room_type.unique()

    def room_type_value_counts(self):
        """
        Counts the occurrences of each unique value in the 'room_type' column of the data.
        
        Returns:
        pd.Series: A Series containing the counts of unique room types.
        """
        return self.data['room_type'].value_counts()

    def bar_plot_neighbourhood(self):
        """
        Creates a horizontal bar plot for the frequency distribution of the 'neighbourhood' column.
        """
        self.data['neighbourhood'].value_counts().sort_values().plot.barh(title="Freq Dist of neighbourhood Indicator", rot=0)

    def bar_plot_room_type(self):
        """
        Creates a horizontal bar plot for the frequency distribution of the 'room_type' column.
        """
        self.data['room_type'].value_counts().sort_values().plot.barh(title="Freq Dist of room_type Indicator", rot=0)

    def price_describe(self):
        """
        Computes descriptive statistics for the 'price' column of the data.
        
        Returns:
        .
        """
        return self.data['price'].describe()

    def minimum_nights_describe(self):
        """
        Computes descriptive statistics for the 'minimum_nights' column of the data.
        
        Returns:
        .
        """
        return self.data['minimum_nights'].describe()
    
    def number_of_reviews_describe(self):
        """
        Computes descriptive statistics for the 'number_of_reviews' column of the data.
        
        Returns:
        .
        """
        return self.data['number_of_reviews'].describe()
    
    def plot_price_distribution(self, figsize=(12, 6)):
        """
        This method plots a box plot and histogram of the 'price' column.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (12, 6).
        
        Returns:
        - None. Displays the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot of 'price'
        self.data['price'].plot.box(ax=axes[0], title='Boxplot of Price',fontsize=12, grid=True)
        axes[0].xaxis.set_visible(False)  # Hide x-axis label
        axes[0].title.set_size(14)  # Set title font size

        # Histogram of 'price'
        self.data['price'].plot.hist(ax=axes[1], title='Histogram of Price', bins=10, ec='black', fontsize=12, grid=True)
        axes[1].title.set_size(14)  # Set title font size

        # Display the plots
        plt.tight_layout()
        plt.show()
        
    def plot_price_distribution_seaborn(self, figsize=(12, 6)):
        """
        This method plots a box plot and histogram of the 'price' column using Seaborn.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (12, 6).
        
        Returns:
        - None. Displays the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot of 'price'
        sns.boxplot(x=self.data['price'], ax=axes[0])
        axes[0].set_title('Boxplot of Price', fontsize=14)
        axes[0].xaxis.set_visible(False)  # Hide x-axis label

        # Histogram of 'price'
        sns.histplot(self.data['price'], bins=10, kde=True, ax=axes[1])
        axes[1].set_title('Histogram of Price', fontsize=14)

        # Display the plots
        plt.tight_layout()
        plt.show()

        
    def plot_minimum_nights_distribution(self, figsize=(12, 6)):
        """
        This method plots a box plot and histogram of the 'minimum_nights' column.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (12, 6).
        
        Returns:
        - None. Displays the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot of 'minimum_nights'
        self.data['minimum_nights'].plot.box(ax=axes[0], title='Boxplot of Minimum Nights', fontsize=12, grid=True)
        axes[0].xaxis.set_visible(False)  # Hide x-axis label
        axes[0].title.set_size(14)  # Set title font size

        # Histogram of 'minimum_nights'
        self.data['minimum_nights'].plot.hist(ax=axes[1], title='Histogram of Minimum Nights', bins=10, ec='black', fontsize=12, grid=True)
        axes[1].title.set_size(14)  # Set title font size

        # Display the plots
        plt.tight_layout()
        plt.show()
        
        
    def plot_minimum_nights_distribution_seaborn(self, figsize=(12, 6)):
        """
        This method plots a box plot and histogram of the 'minimum_nights' column using Seaborn.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (12, 6).
        
        Returns:
        - None. Displays the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot of 'minimum_nights'
        sns.boxplot(x=self.data['minimum_nights'], ax=axes[0])
        axes[0].set_title('Boxplot of Minimum Nights', fontsize=14)
        axes[0].xaxis.set_visible(False)  # Hide x-axis label

        # Histogram of 'minimum_nights'
        sns.histplot(self.data['minimum_nights'], bins=10, kde=True, ax=axes[1])
        axes[1].set_title('Histogram of Minimum Nights', fontsize=14)

        # Display the plots
        plt.tight_layout()
        plt.show()
        
        
    def plot_number_of_reviews_distribution(self, figsize=(12, 6)):
        """
        This method plots a box plot and histogram of the 'number_of_reviews' column.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (12, 6).
        
        Returns:
        - None. Displays the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot of 'number_of_reviews'
        self.data['number_of_reviews'].plot.box(ax=axes[0], title='Boxplot of Number of Reviews', fontsize=12, grid=True)
        axes[0].xaxis.set_visible(False)  # Hide x-axis label
        axes[0].title.set_size(14)  # Set title font size

        # Histogram of 'number_of_reviews'
        self.data['number_of_reviews'].plot.hist(ax=axes[1], title='Histogram of Number of Reviews', bins=10, ec='black', fontsize=12, grid=True)
        axes[1].title.set_size(14)  # Set title font size

        # Display the plots
        plt.tight_layout()
        plt.show()
        
    def plot_number_of_reviews_distribution_seaborn(self, figsize=(12, 6)):
        """
        This method plots a box plot and histogram of the 'number_of_reviews' column using Seaborn.
        
        Parameters:
        - figsize (tuple): Specifies the figure size. Default is (12, 6).
        
        Returns:
        - None. Displays the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Box plot of 'number_of_reviews'
        sns.boxplot(x=self.data['number_of_reviews'], ax=axes[0])
        axes[0].set_title('Boxplot of Number of Reviews', fontsize=14)
        axes[0].xaxis.set_visible(False)  # Hide x-axis label

        # Histogram of 'number_of_reviews'
        sns.histplot(self.data['number_of_reviews'], bins=10, kde=True, ax=axes[1])
        axes[1].set_title('Histogram of Number of Reviews', fontsize=14)

        # Display the plots
        plt.tight_layout()
        plt.show()
        
        
    def group_and_aggregate(self, group_by, agg_column, agg_func):
        """
        Groups the data by specified columns and applies an aggregation function.

        Parameters:
            group_by (list): List of columns to group by.
            agg_column (str): Column to apply the aggregation on.
            agg_func (str): Aggregation function ('mean', 'sum', 'count', etc.)

        Returns:
            A DataFrame with grouped and aggregated data.
        """
        return self.data.groupby(group_by)[agg_column].agg(agg_func)
    

    def create_pivot_table(self, index, columns, values, aggfunc='mean'):
        """
        Creates a pivot table from the DataFrame.

        Parameters:
            index (str or list): Column(s) to use as index in pivot table.
            columns (str or list): Column(s) to use as columns in pivot table.
            values (str): Column to aggregate.
            aggfunc (str or function): Aggregation function to be used.

        Returns:
            A pivot table as a DataFrame.
        """
        return pd.pivot_table(self.data, index=index, columns=columns, values=values, aggfunc=aggfunc)
    
    def melt_data(self, id_vars, value_vars, var_name='variable', value_name='value'):
        """
        Reshapes DataFrame using melt.

        Parameters:
        id_vars (list of str): Columns to keep as identifier variables.
        value_vars (list of str): Columns to be melted.
        var_name (str): Name to assign to the melted variable column.
        value_name (str): Name to assign to the melted value column.

        Returns:
        pd.DataFrame: Reshaped DataFrame.
        """
        return pd.melt(self.data, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    
    
    def plot_pairplot(data, hue=None):
        """
        Generates a pair plot for selected features in the dataset.

        :param data: DataFrame containing the dataset.
        :param hue: The name of the column in 'data' to be used as hue.
        """
        sns.pairplot(data, hue=hue)
        plt.show()
        

    def plot_correlation_heatmap(data):
        """
        Generates a heatmap for the correlation matrix of the dataset.

        :param data: DataFrame containing the dataset.
        """
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True)
        plt.show()
    
    

