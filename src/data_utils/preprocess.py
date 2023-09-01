import pandas as pd

def dropna(dataframe, columns):
    copy_dataframe = dataframe.copy(deep=True)
    return copy_dataframe.dropna(subset=columns, how="all").reset_index(drop=True)

def fillzero(dataframe):
    copy_dataframe = dataframe.copy(deep=True)
    return copy_dataframe.fillna(0)
