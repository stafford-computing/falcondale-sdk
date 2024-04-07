"""
Module implementing generic QNN setups
"""

import logging

from sklearn.svm import SVC

import pennylane as qml
import pennylane.numpy as np

from .kernels import angle_kernel


class QSVC:
    """
    Pennylane based Quantum Support Vector Classifier
    """

    def __init__(self, inputs: int, verbose: bool = False):
        # Num of features
        self.num_qubits = inputs
        self.device = qml.device("default.qubit", wires=self.num_qubits)

        self.qnode = None
        self.model = None

        # Unknown yet
        self.num_classes = 0

        self.verbose = verbose

    def _build_model(self):
        """
        Build the model
        """
        self.qnode = qml.QNode(angle_kernel, self.device)

        def kernel_matrix(A, B):
            """Compute the matrix whose entries are the kernel
            evaluated on pairwise data from sets A and B."""
            return np.array([[self.qnode(a, b) for b in B] for a in A])

        # Create an SVC with a precomputed kernel
        self.model = SVC(
            C=1.0,
            kernel=kernel_matrix,
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=True,
            tol=0.001,
            cache_size=200,
            class_weight="balanced",
            verbose=self.verbose,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,
        )

    def classify(self, feature_vecs):
        """
        Classify the feature vector in one of the classes
        """
        return self.model.predict(feature_vecs)

    def probabilities(self, feature_vecs):
        """
        Classify the feature vector in one of the classes
        """
        return self.model.predict_proba(feature_vecs)

    def accuracy(self, labels, hard_predictions):
        """
        Computes the accuracy given some actuals and predictions
        """
        loss = 0
        for label, p in zip(labels, hard_predictions):
            if abs(label - p) < 1e-5:
                loss = loss + 1
        loss = loss / labels.shape[0]
        return loss

    def fit(self, x_df, y_np):
        """
        Main training function
        """
        logging.info("Training started...")
        self.num_classes = len(np.unique(y_np))

        # build ansatz
        self._build_model()

        # Drop index
        x_df.reset_index(drop=True)
        feat_vec = x_df.to_numpy()

        self.model.fit(feat_vec, y_np)
        predictions = self.model.predict(feat_vec)

        return None, self.accuracy(y_np, predictions)

    def score(self, feat_df, label_df):
        """
        Uses the input dataframe to score predictions
        """
        feat_vec = feat_df.to_numpy()
        y_label = label_df.to_numpy()

        predictions_train = self.classify(feat_vec)
        accuracy = self.accuracy(y_label, predictions_train)

        return accuracy

    def predict(self, feat_df):
        """
        Uses the input dataframe to score predictions
        """
        feat_vec = feat_df.to_numpy()

        return self.classify(feat_vec)

    def predict_proba(self, feat_df):
        """
        Uses the input dataframe to score predictions
        """
        feat_vec = feat_df.to_numpy()

        return self.probabilities(feat_vec)
