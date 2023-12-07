import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class Modeling:
    """
    A class to build and evaluate a machine learning model for predicting Airbnb listing prices.

    Attributes
    ----------
    data : pd.DataFrame
        The entire dataset including the target variable.
    target : str
        The name of the target variable column.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        The seed used by the random number generator.
    model : sklearn estimator object
        The trained model after calling the `train` method.
    X_train : ndarray
        The training data features after calling `prepare_data`.
    X_test : ndarray
        The test data features after calling `prepare_data`.
    y_train : ndarray
        The training data target after calling `prepare_data`.
    y_test : ndarray
        The test data target after calling `prepare_data`.
    pipeline : sklearn Pipeline object
        The complete pipeline including preprocessing and the regressor.
    
    Methods
    -------
    __init__(self, data, target, test_size=0.2, random_state=42)
        Constructor for the Modeling class.
    prepare_data(self)
        Prepares training and testing datasets.
    create_pipeline(self)
        Creates a scikit-learn pipeline for data preprocessing and modeling.
    train(self)
        Trains the model on the training data.
    evaluate(self)
        Evaluates the model's performance on the test set.
    cross_validate(self, cv=5)
        Performs cross-validation to evaluate model performance.
    grid_search(self, param_grid)
        Performs a grid search to find the best hyperparameters for the model.
    plot_residuals(self)
        Plots the residuals of the model predictions.
    plot_actual_vs_predicted(self)
        Visualizes the actual vs predicted prices using the test set.

    """

    def __init__(self, data, target, test_size=0.2, random_state=42):
        """
        Initializes the Modeling object with data and configuration for model building and evaluation.

        Parameters
        ----------
        data : pd.DataFrame
            The entire dataset including features and the target variable.
        target : str
            The name of the target variable column.
        test_size : float, optional
            The proportion of the dataset to include in the test split. Default is 0.2.
        random_state : int, optional
            The seed used by the random number generator. Default is 42.
        """
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
        self.pipeline = self.create_pipeline()

    def prepare_data(self):
        """
        Splits the dataset into training and testing sets, preparing it for the modeling process.

        Returns
        -------
        tuple
            A tuple containing training and testing sets for both features (X) and target (y).
        """
    
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def create_pipeline(self):
        """
        Constructs a pipeline for preprocessing and modeling with specified transformers and a regressor.

        Returns
        -------
        sklearn.pipeline.Pipeline
            A configured pipeline with preprocessing steps and a random forest regressor.
        """
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.X_train.select_dtypes(include=['object', 'bool']).columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=self.random_state))
        ])
        return pipeline

    def train(self):
        """
        Fits the model to the training data using the prepared pipeline.

        Returns
        -------
        None
        """
        self.pipeline.fit(self.X_train, self.y_train)
        self.model = self.pipeline.named_steps['regressor']

    def evaluate(self):
        """
        Evaluates the model's performance on the test set using mean squared error and R squared metrics.

        Returns
        -------
        tuple
            A tuple containing mean squared error and R squared score of the model's predictions.
        """
        predictions = self.pipeline.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        print(f'Mean Squared Error: {mse}')
        print(f'R^2 Score: {r2}')
        return mse, r2

    def cross_validate(self, cv=5):
        """
        Applies cross-validation to the training data to assess model performance.

        Parameters
        ----------
        cv : int, optional
            Number of cross-validation folds. Default is 5.

        Returns
        -------
        numpy.ndarray
            Array of negative mean squared error scores from the cross-validation.
        """
        cv_scores = cross_val_score(self.pipeline, self.X_train, self.y_train, cv=cv, scoring='neg_mean_squared_error')
        print(f'CV Mean Squared Error: {-cv_scores.mean()}')
        return cv_scores

    def grid_search(self, param_grid):
        """
        Executes a grid search over a parameter grid to find the best model parameters.

        Parameters
        ----------
        param_grid : dict
            Dictionary with parameter names as keys and lists of parameter settings to try as values.

        Returns
        -------
        dict
            The best parameter values found during the grid search.
        """
        search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_.named_steps['regressor']
        print(f'Best parameters: {search.best_params_}')
        return search.best_params_
    

    
    def plot_residuals(self):
        """
        Creates a histogram of the residuals to visualize the distribution of prediction errors.

        Returns
        -------
        None
        """
        predictions = self.pipeline.predict(self.X_test)
        residuals = self.y_test - predictions
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, bins=30, kde=True)
        plt.xlabel('Residuals')
        plt.title('Histogram of Residuals')
        plt.show()
    
    def plot_actual_vs_predicted(self):
        """
        Displays a scatter plot comparing actual and predicted prices, along with a line of perfect prediction.

        Returns
        -------
        None
        """
        predictions = self.pipeline.predict(self.X_test)
        plt.figure(figsize=(10,6))
        sns.scatterplot(self.y_test, predictions)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')

        # Calculate the maximum price to define the limits of the diagonal line
        max_price = max(self.y_test.max(), predictions.max())

        # Plot a diagonal line
        plt.plot([0, max_price], [0, max_price], '--k', linewidth=2, label='Perfect Prediction')
    
        # Add legend to the plot to differentiate the actual vs predicted points from the diagonal line
        plt.legend()

        plt.show()



