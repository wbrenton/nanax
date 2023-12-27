import numpy as np
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def expected_improvement(gp, x, y, greater_is_better=True, n_params=2):
    """
    Expected improvement acquisition function using covariance matrix.
    """
    x_to_predict = x.reshape(-1, n_params)

    # Predict using GP and get covariance matrix
    mu, cov = gp.predict(x_to_predict) #return_cov=True

    # Extract variances (diagonal of covariance matrix)
    variances = np.diag(cov)
    sigma = np.sqrt(variances)

    if greater_is_better:
        loss_optimum = np.max(y)
    else:
        loss_optimum = np.min(y)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] = 0.0

    ei = -1 * expected_improvement
    return x[np.argmax(ei)]


if __name__ == "__main__":
    # adapted from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy_targets.html#sphx-glr-download-auto-examples-gaussian-process-plot-gpr-noisy-targets-py
    import matplotlib.pyplot as plt

    # Define the test function (e.g., a sine wave)
    X = np.linspace(start=0, stop=10, num=1_000).reshape(-1, 1)
    X_norm = (X - X.min()) / (X.max() - X.min())
    y = np.squeeze(X * np.sin(X))

    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
    X_train_norm, X_train, y_train = X_norm[training_indices], X[training_indices], y[training_indices]

    gaussian_process.fit(X_train_norm, y_train)

    mean_prediction, std_prediction = gaussian_process.predict(X_norm, return_std=True)

    plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    plt.scatter(X_train, y_train, label="Observations")
    plt.plot(X, mean_prediction, label="Mean prediction")
    plt.fill_between(
        X.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset")
    plt.savefig("gp_norm.png")
    
    
    
# class GaussianProcess(GaussianProcessRegressor):
#     def __init__(
#         self,
#         kernel=None,
#         normalize_X=True,
#         normalize_y=False,
#         *,
#         alpha=1e-10,
#         optimizer="fmin_l_bfgs_b",
#         n_restarts_optimizer=0,
#         copy_X_train=True,
#         n_targets=None,
#         random_state=None,
#     ):
#         super().__init__(
#             kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
#             copy_X_train=copy_X_train, n_targets=n_targets, random_state=random_state
#         )

#         self.normalize_X = normalize_X
#         self.normalize_y = normalize_y
#         self.X_stats = {'min': np.inf, 'max': -np.inf}
#         self.y_stats = {'min': np.inf, 'max': -np.inf}

#     def fit(self, X, y):
#         if self.normalize_X:
#             self.X_stats, X = self._update_stats_and_normalize(self.X_stats, X)
#         if self.normalize_y:
#             self.y_stats, y = self._update_stats_and_normalize(self.y_stats, y)
#         return super().fit(X, y)

#     def predict(self, X, return_std=False, return_cov=False):
#         out = super().predict(X, return_std=return_std, return_cov=return_cov)
#         y_pred, rest = out[0], out[1:]
#         if self.normalize_y:
#             y_pred = self._unnormalize(y_pred, self.y_stats)
#         return (y_pred, *rest)
    
#     def _update_stats_and_normalize(self, stats, array):
#         stats['min'] = np.minimum(array.min(), stats['min'])
#         stats['max'] = np.maximum(array.max(), stats['max'])
#         return stats, (array - stats['min']) / (stats['max'] - stats['min'])
    
#     def _unnormalize(self, array, stats):
#         return array * (stats['max'] - stats['min']) + stats['min']