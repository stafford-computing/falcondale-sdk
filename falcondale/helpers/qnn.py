"""
Module implementing generic QNN setups
"""

import logging

import torch
import torch.optim as optim
from torch.autograd import Variable

import pennylane as qml
import pennylane.numpy as np


class QNN:
    """
    Pennylane based Quantum Neural Network
    """

    def __init__(self, inputs: int, layers: int, verbose: bool = False):
        # Num of features
        self.inputs = inputs

        self.num_qubits = int(np.ceil(np.log2(inputs)))
        self.device = qml.device("default.qubit", wires=self.num_qubits)

        self.layers = layers
        self.qnodes = []

        # Unknown yet
        self.num_classes = 0
        self.params = None

        self.verbose = verbose

    def layer(self, weights):
        """
        Basic layer structure
        """
        for i in range(self.num_qubits):
            qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
        for j in range(self.num_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        if self.num_qubits >= 2:
            # Apply additional CNOT to entangle the last with the first qubit
            qml.CNOT(wires=[self.num_qubits - 1, 0])

    def circuit(self, weights, feat=None):
        """
        Circuit definition (QNN model)
        """
        qml.AmplitudeEmbedding(feat, range(self.num_qubits), pad_with=0.0, normalize=True)

        for W in weights:
            self.layer(W)

        return qml.expval(qml.PauliZ(0))

    def _build_model(self):
        """
        For multiclass we will be building once against
        the rest type of model
        """
        for _ in range(self.num_classes):
            qnode = qml.QNode(self.circuit, self.device)
            self.qnodes.append(qnode)

    def variational_classifier(self, q_circuit, params, feat):
        """
        Model output
        """
        weights = params[0]
        bias = params[1]
        return q_circuit(weights, feat=feat) + bias

    def multiclass_loss(self, feature_vecs, true_labels):
        """
        Multiclass loss calculation
        """
        loss = 0
        num_samples = len(true_labels)
        for i, feature_vec in enumerate(feature_vecs):
            # Compute the score given to this sample by the classifier corresponding to the
            # true label. So for a true label of 1, get the score computed by classifer 1,
            # which distinguishes between "class 1" or "not class 1".
            s_true = self.variational_classifier(
                self.qnodes[int(true_labels[i])],
                (
                    self.params[0][int(true_labels[i])],
                    self.params[1][int(true_labels[i])],
                ),
                feature_vec,
            )
            s_true = s_true.float()
            li = 0

            # Get the scores computed for this sample by the other classifiers
            for j in range(self.num_classes):
                if j != int(true_labels[i]):
                    s_j = self.variational_classifier(
                        self.qnodes[j],
                        (self.params[0][j], self.params[1][j]),
                        feature_vec,
                    )
                    s_j = s_j.float()
                    li += torch.max(torch.zeros(1).float(), s_j - s_true)
            loss += li

        return loss / num_samples

    def classify(self, feature_vecs):
        """
        Classify the feature vector in one of the classes
        """
        predicted_labels = []
        for _, feature_vec in enumerate(feature_vecs):
            scores = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                score = self.variational_classifier(self.qnodes[c], (self.params[0][c], self.params[1][c]), feature_vec)
                scores[c] = float(score)
            pred_class = np.argmax(scores)
            predicted_labels.append(pred_class)
        return predicted_labels

    def probabilities(self, feature_vecs):
        """
        Classify the feature vector in one of the classes
        """
        predicted_scores = []
        for _, feature_vec in enumerate(feature_vecs):
            scores = np.zeros(self.num_classes)
            for c in range(self.num_classes):
                score = self.variational_classifier(self.qnodes[c], (self.params[0][c], self.params[1][c]), feature_vec)
                scores[c] = float(score)
            predicted_scores.append(scores)
        return np.asarray(predicted_scores)

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

    def fit(self, x_df, y_np, max_iterations: int = 20, early_stop: float = 0.98):
        """
        Main training function
        """
        logging.info("Training started...")
        num_train = len(x_df)
        batch_size = int(num_train / 10)
        self.num_classes = len(np.unique(y_np))

        # build ansatz
        self._build_model()

        # Drop index
        x_df.reset_index(drop=True)

        # Initialize the parameters
        all_weights = [
            Variable(0.1 * torch.randn(self.layers, self.num_qubits, 3), requires_grad=True)
            for _ in range(self.num_classes)
        ]
        all_bias = [Variable(0.1 * torch.ones(1), requires_grad=True) for _ in range(self.num_classes)]
        optimizer = optim.Adam(all_weights + all_bias, lr=0.01)
        self.params = (all_weights, all_bias)

        costs, train_acc = [], []
        # train the variational classifier
        for iteration in range(max_iterations):
            # While df
            batch_index = np.random.randint(0, num_train, (batch_size,))
            feat_vec = x_df.to_numpy()[batch_index, :]
            y_train = y_np[batch_index]

            optimizer.zero_grad()
            curr_cost = self.multiclass_loss(feat_vec, y_train)
            curr_cost.backward()
            optimizer.step()

            # Compute predictions on train and validation set
            predictions_train = self.classify(feat_vec)
            acc_train = self.accuracy(y_train, predictions_train)

            print(
                "Iter: {:5d} | Cost: {:0.7f} | Acc train: {:0.7f}" "".format(iteration + 1, curr_cost.item(), acc_train)
            )

            costs.append(curr_cost.item())
            train_acc.append(acc_train)

            # Early stop
            if acc_train > early_stop:
                break

        return costs, train_acc

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
