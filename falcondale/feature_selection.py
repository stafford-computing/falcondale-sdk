"""
This module focuses on feature selection methods to diminish the dimensionality of a target dataset.

"""
import dimod
import numpy as np
import itertools

# Problem modelling imports
from qiskit.result import QuasiDistribution
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils.algorithm_globals import algorithm_globals
from qiskit.primitives import Sampler

from .data import Dataset
from .solvers import dwave_solver, neal_solver
from .helpers.mutual_information import \
    conditional_mutual_information, mutual_information, prob

def _compose_bqm(input_ds:Dataset, max_cols:int = None):
    """
    Builds the QUBO required for QFS
    """

    pauli_list = []
    col_list = [feat for feat in input_ds.columns() if feat != input_ds.target]

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # Importance
    for i, column in enumerate(col_list):
        minf = mutual_information(
            prob(input_ds.select([input_ds.target, column]).values), 1)
        bqm.add_variable(column, -minf)
        pauli_list.append(("Z", [i], -minf))

    # Redundancy
    for field0, field1 in itertools.combinations(col_list, 2):
        cmi_01 = conditional_mutual_information(
            prob(input_ds.select([input_ds.target, field0, field1]).values), 1, 2)
        cmi_10 = conditional_mutual_information(
            prob(input_ds.select([input_ds.target, field1, field0]).values), 1, 2)
        bqm.add_interaction(field0, field1, -cmi_01)
        bqm.add_interaction(field1, field0, -cmi_10)

        i = col_list.index(field0)
        j = col_list.index(field1)
        pauli_list.append(("ZZ", [i, j], -cmi_01))
        pauli_list.append(("ZZ", [j, i], -cmi_10))

    # Penalty
    if max_cols & max_cols < len(col_list):
        bqm.update(dimod.generators.combinations(bqm.variables, max_cols, strength=10*max_cols))

    bqm.normalize()     # scale the BQM to (-1, 1) biases

    op = SparsePauliOp.from_sparse_list(
        pauli_list,
        num_qubits=len(col_list)
    )

    return bqm, op

def _sample_most_likely(state_vector):
    """Compute the most likely binary string from state vector.
    Args:
        state_vector: State vector or quasi-distribution.

    Returns:
        Binary string as an array of ints.
    """
    if isinstance(state_vector, QuasiDistribution):
        values = list(state_vector.values())
    else:
        values = state_vector
    n = int(np.log2(len(values)))
    k = np.argmax(np.abs(values))
    x = [int(digit) for digit in np.binary_repr(n,k)]
    x.reverse()
    return np.asarray(x)

def qfs_sim_qaoa(input_ds:Dataset, max_cols:int = None):
    """ Compose the problem to run on a locally simulated QAOA setup"""

    bqm, qubit_op = _compose_bqm(input_ds, max_cols)

    qubo, _ = bqm.to_qubo()

    algorithm_globals.random_seed = 12345
    optimizer = COBYLA()
    qaoa = QAOA(Sampler(), optimizer, reps=2)

    result = qaoa.compute_minimum_eigenvalue(qubit_op)
    col_mask = _sample_most_likely(result.eigenstate)

    col_list = []
    for mask, column in zip(col_mask, input_ds._columns):
        if mask == 1:
            col_list.append(column)

    return col_list

def qfs(token:str, input_ds:Dataset, max_cols:int = None) -> list[str]:
    """
    Implements a QUBO problem so that the set of features to be used can be implemented on
    a quantum annealer.

    Ref:
    - https://dl.acm.org/citation.cfm?id=2623611
    - https://arxiv.org/pdf/2203.13261.pdf

    Parameters:
    token (str) : Token for the DWave connectivity
    input (DataFrame): receives a pandas Dataframe

    Return:
    DataFrame: outputs the transformed dataset
    """

    bqm, _ = _compose_bqm(input_ds, max_cols)

    # Send it to DWave's device
    samples = dwave_solver(bqm, token)
    selection = samples.first.sample

    col_list = []
    for column in selection:
        if selection[column] == 1:
            col_list.append(column)

    return col_list

def qfs_sim(input_ds:Dataset, max_cols:int = None) -> list[str]:
    """
    Implements a QUBO problem so that the set of features to be used can be implemented on
    a quantum simulated annealer.

    Ref:
    - https://dl.acm.org/citation.cfm?id=2623611
    - https://arxiv.org/pdf/2203.13261.pdf

    Parameters:
    token (str) : Token for the DWave connectivity
    input (DataFrame): receives a pandas Dataframe

    Return:
    DataFrame: outputs the transformed dataset
    """

    bqm, _ = _compose_bqm(input_ds, max_cols)

    # Send it to DWave's device
    samples = neal_solver(bqm)
    selection = samples.first.sample

    col_list = []
    for column in selection:
        if selection[column] == 1:
            col_list.append(column)

    return col_list
