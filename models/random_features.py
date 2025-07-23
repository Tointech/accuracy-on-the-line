from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression


def get_model(params=None):
    params = params or {}
    gamma = params.get("gamma", 1.0)
    n_components = params.get("n_components", 500)
    lr_params = params.get("lr_params", {})
    return make_pipeline(
        StandardScaler(),
        RBFSampler(gamma=gamma, n_components=n_components),
        LogisticRegression(**lr_params),
    )
