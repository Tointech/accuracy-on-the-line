from sklearn.linear_model import LogisticRegression


def get_model(params=None):
    params = params or {}
    return LogisticRegression(**params)
