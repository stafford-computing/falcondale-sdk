import pennylane as qml
import pennylane.numpy as np


def angle_kernel(x1, x2):
    """
    Angle encoding using Kernel implementation
    """
    num_qubits = len(x1)

    projector = np.zeros((2**num_qubits, 2**num_qubits))
    projector[0, 0] = 1

    # ComputeUncompute pattern
    qml.AngleEmbedding(x1, wires=range(num_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(num_qubits))

    return qml.expval(qml.Hermitian(projector, wires=range(num_qubits)))
