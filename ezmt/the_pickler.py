import json
import logging
import os
import time

import pandas as pd

from ezmt.common_funcs import is_picklable

_log = logging.getLogger("ezmt.pickler")


class ThePickler(json.JSONEncoder):
    def __init__(self, folder="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.folder = folder
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            # if object is of type dataframe or series, save to a csv
            file_name = f"{id(obj)}_{int(time.time()*1000)}"
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                file_name += ".csv"
                path = os.path.join(self.folder, file_name)
                obj.to_csv(path)
                return file_name
            # otherwise try to pickle object
            import pickle

            try:
                pickled_data = pickle.dumps(obj)
                file_name += ".pkl"
                with open(os.path.join(self.folder, file_name), "wb") as f:
                    f.write(pickled_data)
                return file_name
            except Exception as e:
                _log.error("Error pickling object: %s", e)
                return str(obj)


def check_state_picklability(state):
    if not is_picklable(state):
        for k, v in state.items():
            if not is_picklable(v):
                raise ValueError(f'Cannot pickle non-picklable value: {k}={v}')
