import pandas as pd
from fast_ml.outlier_treatment import OutlierTreatment

from logger import Logger


class DataCleaner:
    """
    Class for data cleaning

    """

    def __init__(self):
        """
        Initilize data cleaner class.

        """

        self.logger = Logger("data_cleaner").get_app_logger()

    def missing_percentage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the missing percentage value in each columns

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            df: pd.DataFrame
                a pandas dataframe of all the columns and their respective missing value percentage


        """

        percent_missing = df.isnull().sum() * 100 / len(df)
        missing_value_df = pd.DataFrame(
            {"column_name": df.columns, "percent_missing": percent_missing}
        )
        missing_value_df.reset_index(drop=True, inplace=True)
        self.logger.info("Missing values percentage returned")
        return missing_value_df

    def check_number_of_duplicates(self, df: pd.DataFrame) -> int:
        """
        Returns the total number of duplicate rows in a dataframe

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            count: int
                the total number of number of duplicate rows in a dataframe


        """

        self.logger.info("Duplicate count returned")
        return df.duplicated().sum()

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a dataframe with its outliers removed

        Parameters:
        ---
            df: pd.DataFrame
                a pandas dataframe as an input

        Returns:
        ---
            df: pd.DataFrame
                a pandas dataframe with treated outliers


        """

        df_copy = df.copy()
        self.logger.info("Outliers treated")
        return OutlierTreatment(method="iqr").fit(df_copy).transform(df_copy)
