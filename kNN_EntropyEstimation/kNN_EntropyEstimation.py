import numpy as np
import scipy.special


def psi(i):
    """
    Digamma function for convenience
    """
    return scipy.special.digamma(i)


def c(d):
    """
    Volume of the d-dimensional unit ball divided by 2**d
    """
    return np.pi**(d / 2) / scipy.special.gamma(1 + d / 2) / 2**d


def estimate_entropy(samples, k, method='maxNorm', fac=1.):
    """
    Implements the estimators II.A ('euclidean') and II.B ('maxNorm') from PhysRevE.69.066138.

        Arguments:

            * ``samples``: The samples of the distribution of which an entropy estimate is desired, with shape (N, dim).
            * ``k``: Hyperparameter. The k-th NN is used for the distance.
            * ``method``: Use either II.A or II.B from the referenced paper.
            * ``fac``: Correction factor if the distribution has normalization unequal 1.

        Returns:
            The estimated entropy.
    """
    N = samples.shape[0]
    dim = samples.shape[1]
    corr = samples[None, ...] - samples[:, None, :]

    if method == 'euclidean':
        corr_dist = np.linalg.norm(corr, axis=-1)
        eps = 2 * np.partition(corr_dist, k, axis=-1)[:, k]

        entropy_estimate = dim * np.mean(np.log(eps / fac))
        entropy_estimate = entropy_estimate - psi(k) + psi(N) + np.log(c(dim))

    elif method == 'maxNorm':
        eps = 2 * np.partition(np.max(np.abs(corr), axis=-1), k, axis=-1)[:, k]
        entropy_estimate = dim * np.mean(np.log(eps / fac), axis=-1)
        entropy_estimate = entropy_estimate - psi(k) + psi(N)

    return entropy_estimate


if __name__ == '__main__':
    N = 1000
    dim = 2
    k = 2
    runs = 100

    res_euclid = []
    res_maxNorm = []
    for run in range(runs):
        print(f"run no. {run} / {runs}")
        samples = np.random.normal(size=(N, dim))
        res_maxNorm.append(estimate_entropy(samples, k, method='maxNorm'))
        res_euclid.append(estimate_entropy(samples, k, method='euclidean'))

    entropy_analytical = 0.5 * dim * np.log(2 * np.pi * np.exp(1))
    print(f"analytical: {entropy_analytical}")
    print(f"maxNorm : {np.mean(res_maxNorm)} +- {np.std(res_maxNorm)}")
    print(f"euclid mean: {np.mean(res_euclid)} +- {np.std(res_euclid)}")
