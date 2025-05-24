import pandas as pd

from datetime import datetime

def get_min_diff(
    data: pd.DataFrame,
    date_o_colname: str,
    date_i_colname: str
) -> pd.Series:
    """
        Get difference in minutes between two date columns

        Args:
            data (pd.DataFrame): dataframe with date columns
            date_o_colname (str): name of flight date column
            date_i_colname (str): name of scheduled flight date column
        Returns:
            pd.Series: difference in minutes of the two date columns.
    """
    
    fecha_o = datetime.strptime(data[date_o_colname], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data[date_i_colname], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds())/60
    return min_diff