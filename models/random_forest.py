from sklearn.ensemble import RandomForestClassifier


def get_model(params=None):
    params = params or {}
    return RandomForestClassifier(**params)
