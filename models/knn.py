from sklearn.neighbors import KNeighborsClassifier


def get_model(params=None):
    params = params or {}
    return KNeighborsClassifier(**params)
