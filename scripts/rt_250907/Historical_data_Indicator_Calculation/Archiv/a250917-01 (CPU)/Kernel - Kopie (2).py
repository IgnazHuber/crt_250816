import os
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


def Kernel(local_extrema):
    local_extrema = np.array(local_extrema)

    # If all values are zero, return a dictionary with zero values
    if np.all(local_extrema == 0):
        return {p: 0 for p in [0.1, 10, 17.5, 25, 35, 45, 50, 57.5, 66.66, 75, 80, 87.5, 90, 95, 97.5, 99.999]}
        # return {p: 0 for p in [0.1, 5, 10, 17.5, 25, 33.1, 35, 40, 45, 50, 57.5, 66.66, 75, 80, 85, 87.5, 92.5, 95, 97.5, 99.99]}

    # Handle the case where all values are the same (but nonzero)
    if np.all(local_extrema == local_extrema[0]):
        mean = local_extrema[0]  # All values are the same
        std = 1e-6  # Small nonzero std to avoid division errors
    else:
        mean = np.mean(local_extrema)
        std = np.std(local_extrema)

    # Percentiles for support levels
    percentiles = [0.1, 10, 17.5, 25, 35, 45, 50, 57.5, 66.66, 75, 80, 87.5, 90, 95, 97.5, 99.999]

    # percentiles = [0.1, 5, 10, 17.5, 25, 33.1, 35, 40, 45, 50, 57.5, 66.66, 72.5, 75, 80, 85, 87.5, 92.5, 95, 97.5, 99.99]

    # Compute support levels using normal distribution
    support_levels = {p: round(norm.ppf(p / 100, mean, std), 4) for p in percentiles}

    # Ensure KDE can handle constant values
    if np.std(local_extrema) == 0:
        return support_levels  # Return normal distribution results only

    # Optional: GPU-accelerated KDE via cuML (if available and enabled)
    if os.environ.get('USE_GPU') == '1':
        try:
            import cupy as cp
            from cuml.neighbors import KernelDensity as CuKDE
            arr = cp.asarray(local_extrema).reshape(-1, 1)
            kde_gpu = CuKDE(kernel='gaussian', bandwidth=1.0)
            kde_gpu.fit(arr)
            x_values_gpu = cp.linspace(float(local_extrema.min()), float(local_extrema.max()), 10000)
            log_dens_gpu = kde_gpu.score_samples(x_values_gpu.reshape(-1, 1))
            dens_gpu = cp.exp(log_dens_gpu)
            cdf_gpu = cp.cumsum(dens_gpu) / cp.sum(dens_gpu)
            result = {}
            for p in percentiles:
                idx = int(cp.argmax(cdf_gpu >= (p / 100.0)))
                result[p] = round(float(x_values_gpu[idx]), 4)
            return result
        except Exception:
            # Fallback to CPU path if GPU path is not available/functional
            pass

    # CPU: Fit a Kernel Density Estimate (KDE)
    kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(local_extrema[:, None])

    # Generate x-values for KDE
    x_values = np.linspace(min(local_extrema), max(local_extrema), 10000)

    # Compute log density estimate
    log_dens_kde = kde.score_samples(x_values[:, None])
    dens_kde = np.exp(log_dens_kde)  # Convert to density

    # Compute cumulative distribution function (CDF)
    cdf_kde = np.cumsum(dens_kde) / np.sum(dens_kde)

    # Determine the percentiles from KDE and round values to 4 decimal places
    support_levels_kde = {p: round(x_values[np.argmax(cdf_kde >= p / 100)], 4) for p in percentiles}

    return support_levels_kde


