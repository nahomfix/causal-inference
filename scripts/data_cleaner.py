import pandas as pd
from fast_ml import eda
from fast_ml.outlier_treatment import OutlierTreatment


class DataCleaner:
    def __init__(self):
        """
        Initilize data cleaner class.

        """

        pass

    def summary_info(self, df: pd.DataFrame) -> pd.DataFrame:
        return eda.df_info(df)

    def remove_outlier(self, df: pd.DataFrame) -> None:
        OutlierTreatment(method="iqr").fit(df).transform(df)
