"""
Aiming for clustering dataset samples using classical, quantum-inspired and quantum techniques.
"""

from .data import Dataset

import numpy as np
from apqc.pqc import PQC
from apqc.density_estimation import DensityEstimator


def prob_q_clustering(input_ds: Dataset, batch: int = 100, forced_sigma: float = None):
    """
    Accelerated Probabilistic Quantum Clustering

    Parameters:
    input_ds (Dataset): Entry dataset
    """
    pqc = PQC(data_gen=input_ds.get_features(), float_type=32, batch=batch, force_cpu=True)

    if forced_sigma:
        pqc.set_sigmas(sigma_value=forced_sigma)
    else:
        # Fit
        d_e = DensityEstimator(data_gen=pqc.data_gen, batch=batch, scale=pqc.scale)
        init_log_sigmas = np.ones((d_e.data_gen.shape[0], 1)) * np.log(1)
        _ = d_e.fit(preset_init=init_log_sigmas)

        log_sigmas = d_e.log_sigma.value()
        sigmas = np.exp(log_sigmas)
        # Fit
        pqc.set_sigmas(sigma_value=float(sigmas[-1]))

    # Train on existing sigmas
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()

    best_solution_key = sorted(
        [(k, v["loglikelihood"]) for k, v in pqc.basic_results.items()],
        key=lambda x: x[1],
    )[0][0]
    best_proba_labels = pqc.basic_results[best_solution_key]["proba_labels"]
    label_probability = pqc.basic_results[best_solution_key]["proba_winner"]

    return best_proba_labels, label_probability
