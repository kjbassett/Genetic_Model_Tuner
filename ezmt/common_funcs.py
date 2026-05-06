import pickle


def resolve_log_states(log_states):
    if log_states is True:
        return None  # None means log all
    if log_states is False:
        return set()
    if isinstance(log_states, int):
        return {log_states}
    return set(log_states)


def is_picklable(obj):
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError):
        return False
