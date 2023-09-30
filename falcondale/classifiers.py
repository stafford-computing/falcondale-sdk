"""
Module implementing the classification methods
"""
from .helpers.qnn import QNN
from .data import Dataset

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from qiskit_aer import AerSimulator
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms import QSVC


class TooManyQubitsNeeded(Exception):
    """
    Current devices cannot go beyond a defined
    number of qubits
    """
    max_qubits = 20

    def __init__(self, qubits, *args):
        super().__init__(args)
        self.max_qubits = qubits

    def __str__(self):
        return f'Too many qubits required {self.max_qubits}'


def _get_metrics(y_true, y_pred) -> dict:

    tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()

    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)

    metrics = {
        "sensitivity": tpr,
        "recall": tnr,
        "precision": tp/(tp+fp),
        "f1": (2*tp)/(2*tp+fn+fp),
        "accuracy": (tp+tn)/(tn+tp+fn+fp),
        "balanced accuracy" : (tpr+tnr)/2
    }

    return metrics

def eval_svc(dataset: Dataset, test_size:float = 0.3):
    """
    Dataset classification using classical SVC method
    """
    x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=test_size)

    # Instantiate the SVC
    svc = SVC(probability=True)

    # Training
    svc.fit(x_train, y_train)

    # Testing
    y_pred = svc.predict(x_test)
    y_pred_proba = svc.predict_proba(x_test)[:, -1]
    metrics = _get_metrics(y_test,y_pred)
    metrics["auc"] = roc_auc_score(y_test, y_pred_proba)

    return svc, classification_report(y_test, y_pred), metrics

def qnn(dataset: Dataset, test_size:float = 0.3, layers: int = 3):
    """
    Dataset classification using QNN method
    """
    x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=test_size)
    size = len(x_train.columns)

    # Instantiate the SVC
    qnn = QNN(
        inputs=size,
        layers = layers
    )

    # Training
    qnn.fit(x_train, y_train)

    # Testing
    y_pred = qnn.predict(x_test)
    y_pred_proba = qnn.predict_proba(x_test)[:, -1]
    metrics = _get_metrics(y_test,y_pred)
    metrics["auc"] = roc_auc_score(y_test, y_pred_proba)

    return qnn, classification_report(y_test, y_pred), metrics


def qsvc(dataset: Dataset, test_size:float = 0.3):
    """
    Dataset class should already provide a dataset with dimensions
    that can fit into the circuit.
    """

    # If too big
    (_, cols) = dataset.size()
    if cols > TooManyQubitsNeeded.max_qubits:
        raise TooManyQubitsNeeded(cols)

    # Magic number
    regularization_constant = 100

    # Defining backend and feature map to be used
    backend = QuantumInstance(
        AerSimulator(method='statevector')
    )

    # ZZ feature map
    feature_map = ZZFeatureMap(feature_dimension=cols, reps=2)

    # Defining quantum kernel and qsvc
    qkernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)
    qsvc = QSVC(quantum_kernel=qkernel, C=regularization_constant, probability=True)

    # Data splitting
    x_train, x_test, y_train, y_test = dataset.train_test_split(test_size=test_size)

    # Training
    qsvc.fit(x_train, y_train)

    # Testing
    y_pred = qsvc.predict(x_test)
    y_pred_proba = qsvc.predict_proba(x_test)[:, -1]
    metrics = _get_metrics(y_test,y_pred)
    metrics["auc"] = roc_auc_score(y_test, y_pred_proba)

    return qsvc, classification_report(y_test, y_pred), metrics
