import pandas as pd

from EZMT import ModelTuner
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, Lasso
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

from multiprocessing import freeze_support


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


def MICE(x_train, x_test, max_iter, tolerance):
    imputer = IterativeImputer(
        max_iter=max_iter,
        tol=tolerance,
        random_state=42,
        estimator=RandomForestRegressor(),
        # verbose=True
    )
    imputer.fit(x_train)
    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)
    return x_train, x_test


def KNNImpute(train_data, test_data, k):
    imputer = KNNImputer(n_neighbors=k)
    imputer.fit(train_data)

    train_data_imputed = imputer.transform(train_data)
    test_data_imputed = imputer.transform(test_data)

    return train_data_imputed, test_data_imputed


def ord_transform(encoder, values):
    try:
        return encoder.transform(values)
    except ValueError:
        if hasattr(encoder, 'classes_'):
            encoder.fit(np.append(encoder.classes_, values))
        else:
            encoder.fit(values)
        return ord_transform(encoder, values)


def ordinal_encode(data, cols=None, encoders=None):
    if cols is None:
        cols = [col for col in data.columns if data[col].dtype == 'object']
    elif isinstance(cols, str):
        cols = [cols]
    if encoders is None:
        encoders = {}

    for col in cols:
        if col in encoders:
            le = encoders[col]
        else:
            le = LabelEncoder()
        data[col] = ord_transform(le, data[col])
        print('Are the mapping being updated in this scope or only in ord_transform?')
        encoders[col] = le  # Save the label encoder to decode later
    return data, encoders


def one_hot_encode(data, cols=None):
    if cols is None:
        cols = [col for col in data.columns if data[col].dtype == 'object']
    elif isinstance(cols, str):
        cols = [cols]
    encoders = {}
    for col in cols:
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(data[[col]])
        encoders[col] = encoder
        new_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
        data[new_cols] = encoded
        data.drop(col, axis=1, inplace=True)
    return data, encoders


def oh_v_or(data, *args, cols=None):
    if cols is None:
        cols = [col for col in data.columns if data[col].dtype == 'object']
    elif isinstance(cols, str):
        cols = [cols]

    if len(args) != len(cols):
        raise Exception('args must same same length as cols')

    encoders = {}
    for i, col in enumerate(cols):
        if args[i]:  # 1
            _ = ordinal_encode(data, col)
        else:
            _ = one_hot_encode(data, col)

        data = _[0]
        encoders[col] = _[1][col]

    return data, encoders


def main():
    freeze_support()

    df = pd.read_csv('G:\\Programming\\mushroom.csv')
    df, encoders = oh_v_or(df, *[1 if col == 'class' else 0 for col in df.columns])

    model_space = [
        # {'func': oh_v_or, 'inputs': ['x_train', 'x_test'], 'outputs':['x_train', 'x_test'], 'args': [1 if col == 'class' else [0, 1] for col in df.columns]},
        {'func': simulate_missing_data, 'name': 'smd', 'inputs': 'x_train', 'outputs': 'x_train', 'args': [(0.1, 0.95)]},
        {'func': simulate_missing_data, 'name': 'smd', 'inputs': 'x_test', 'outputs': 'x_test', 'args': [(0.1, 0.95)]},
        [  # Imputation
            {'func': MICE, 'inputs': ['x_train', 'x_test'], 'outputs': ['x_train', 'x_test'], 'args': [range(1, 6), (0.1, 3)]},
            {'func': KNNImpute, 'inputs': ['x_train', 'x_test'], 'outputs': ['x_train', 'x_test'], 'args': [range(3, 10)]}
        ],
        [
            {'func': RandomForestRegressor, 'name': 'rf', 'outputs': 'model', 'args': [range(10, 50)]},
            {'func': Ridge, 'name': 'lr', 'outputs': 'model', 'kwargs': {'alpha': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}},
            {'func': Lasso, 'name': 'lr', 'outputs': 'model', 'kwargs': {'alpha': [0.0001, 0.001, 0.1, 1, 10, 100, 1000]}}# todo try making prediction model before model space
        ],
        {'func': 'model.fit', 'inputs': ['x_train', 'y_train'], 'outputs': 'model'},
        {'func': 'model.predict', 'inputs': 'x_test', 'outputs': 'test_pred'},
        {'func': mse, 'inputs': ['test_pred', 'y_test'], 'outputs':'score'}
    ]

    mt = ModelTuner(model_space, df, 'class', generations=35, pop_size=6, goal='min')

    model = mt.run()
    print(model)
    print(model.knowledge.keys())
    print(model.score)
    print(model.fitness)

if __name__ == '__main__':
    main()

