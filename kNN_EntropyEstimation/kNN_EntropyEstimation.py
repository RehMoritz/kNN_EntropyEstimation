import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def plot(res, target):

    fig, ax = plt.subplots(2, figsize=(6, 6))

    ax[0].errorbar(np.arange(res.shape[0]), res[..., 0] - target, yerr=res[..., 1],
                   fmt='x')
    ax[0].hlines(y=0, xmin=0, xmax=res.shape[0], linestyle='--', color='black')
    ax[0].grid()
    ax[0].set_xlabel('Id.')
    ax[0].set_ylabel('Diff')

    sigmas = np.arange(0, 3.1, 0.1)
    width = sigmas[1] - sigmas[0]
    x = np.arange(0, 3, 1e-2)
    probs = []
    for sigma in sigmas:
        probs.append(np.mean(np.abs(res[..., 0] - target) / res[..., 1] < sigma))
    ax[1].bar(sigmas - width, probs, width=width, alpha=0.3, align='edge')
    ax[1].plot(x[:-1], scipy.special.erf(x[:-1] / np.sqrt(2)))
    ax[1].grid()
    ax[1].set_xlabel(r'$r/\sigma$')
    ax[1].set_ylabel(r'$p$')

    plt.tight_layout()


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
        entropy_estimate = -psi(k) + psi(N) + np.log(c(dim))

    elif method == 'maxNorm':
        corr_dist = np.max(np.abs(corr), axis=-1)
        entropy_estimate = -psi(k) + psi(N)

    eps = 2 * np.partition(corr_dist, k, axis=-1)[:, k]
    entropy_estimate = entropy_estimate + dim * np.log(eps / fac)

    return np.mean(entropy_estimate), np.std(entropy_estimate) / np.sqrt(N)


if __name__ == '__main__':
    np.random.seed(1)
    N = 700
    dim = 2
    k = 1
    runs = 100

    res_euclid = []
    res_maxNorm = []
    for run in range(runs):
        print(f"run no. {run} / {runs}")
        samples = np.random.normal(size=(N, dim))
        res_maxNorm.append(estimate_entropy(samples, k, method='maxNorm'))
        res_euclid.append(estimate_entropy(samples, k, method='euclidean'))
    res_maxNorm = np.array(res_maxNorm)
    res_euclid = np.array(res_euclid)

    entropy_analytical = 0.5 * dim * np.log(2 * np.pi * np.exp(1))
    print(f"analytical: {entropy_analytical}")
    print(f"maxNorm : {np.mean(res_maxNorm[:, 0])} +- {np.sqrt(np.mean(res_maxNorm[:, 1]**2)) / np.sqrt(runs)}")
    print(f"maxNorm : {np.mean(res_euclid[:, 0])} +- {np.sqrt(np.mean(res_euclid[:, 1]**2)) / np.sqrt(runs)}")

    plot(res_maxNorm, entropy_analytical)
    plt.show()
