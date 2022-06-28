import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fast_ml.eda import df_info, numerical_plots


class DataVisualizer:
    """
    Class for data visualizing

    """

    def __init__(self):
        """
        Initilize data visualizing class.

        """

    def summary_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a summary of information

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            df: pd.DataFrame
                a pandas dataframe of a summary of information


        """

        summary_df = df_info(df)
        return summary_df

    def display_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a correlations table

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            df: pd.DataFrame
                a pandas dataframe of a correlations table


        """

        return df.corr()

    def plot_distributions(
        self, df: pd.DataFrame, numeric_columns: list
    ) -> None:
        """
        Plots histograms of all the numeric columns

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        """

        numerical_plots(df, numeric_columns)

    def plot_pie_chart(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Plots a pie chart of the categorical column

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        """

        palette_color = sns.color_palette("dark")
        df[column_name].value_counts().plot(
            kind="pie", colors=palette_color, autopct="%.0f%%"
        )
        plt.show()

    def plot_correlations(self, df: pd.DataFrame) -> None:
        """
        Plots a heatmap of the correlation matrix

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        """

        plt.figure(figsize=(20, 10))
        mask = np.triu(np.ones_like(df.corr(), dtype=bool))
        heatmap = sns.heatmap(
            df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap="BrBG"
        )
        heatmap.set_title("Correlation Heatmap")
        plt.show()
