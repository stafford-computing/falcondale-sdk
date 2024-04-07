"""
Takes care of the interface with the solvers
"""

import time
import neal
from dwave.cloud import Client


def dwave_solver(problem, token: str):
    """
    Uses D-Wave annealer to solve the given problem
    """
    # Connect using the default or environment connection information
    with Client.from_config(token=token) as client:
        qpu = client.get_solver(name="hybrid_binary_quadratic_model_version2")
        sampler = qpu.sample_bqm(problem, label="Falcondale QFS", time_limit=10)

        # Wait until it finishes
        while not sampler.done():
            time.sleep(5)

        result = sampler.result()

        return result["sampleset"]


def neal_solver(problem):
    """
    Uses D-Wave annealer to solve the given problem
    """
    # Connect using the default or environment connection information
    qpu = neal.SimulatedAnnealingSampler()
    sampleset = qpu.sample(problem)

    return sampleset
