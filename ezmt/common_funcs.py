import pickle


def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False
