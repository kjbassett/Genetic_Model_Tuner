import pandas as pd
from EZMT import ModelTuner
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np

sys.path.append('C:\\Users\\Ken\\Dropbox\\Programming\\EZ-Neural-Net')
from EZNN import create_model

#df = pd.read_csv('G:\\Programming\\mushroom.csv')

mt = ModelTuner(None, df, 'class', generations=5, pop_size=20, goal='min')


def simulate_missing_data(data, chance):
    """
    Simulates missing data in a pandas dataframe by randomly removing a percentage of cells.

    Args:
    - df: pandas dataframe to simulate missing data for
    - chance: float between 0 and 1 representing the probability that a cell will be removed

    Returns:
    - df_miss: pandas dataframe with randomly missing values
    """
    np.random.seed(42)  # set seed for reproducibility
    df_miss = data.copy()  # create a copy of the original dataframe
    for i in range(df_miss.shape[0]):
        for j in range(df_miss.shape[1]):
            if np.random.rand() < chance:
                df_miss.iat[i, j] = np.nan  # randomly set value to NaN
    return df_miss


def ordinal_encode(data, columns=None):
    if columns is None:
        columns = data.columns

    label_encoders = dict()
    for col in columns:
        le = LabelEncoder()
        le.fit(data[col].unique())
        label_encoders[col] = le  # Save the label encoder to decode later
        data[col] = le.transform(data[col])
    return data


def one_hot_encode(data, cols):
    encoders = {}
    for col in cols:
        encoder = OneHotEncoder(sparse=False)
        encoded = encoder.fit_transform(data[[col]])
        encoders[col] = encoder
        new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
        data[new_cols] = encoded
        data.drop(col, axis=1, inplace=True)
    return data, encoders


def one_hot(data, columns=None):
    # don't do numerical columns
    pass



