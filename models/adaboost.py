from sklearn.ensemble import AdaBoostClassifier


def get_model(params=None):
    params = params or {}
    return AdaBoostClassifier(**params)
