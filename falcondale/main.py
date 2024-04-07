# -*- coding: utf-8 -*-
"""
This is Falcondale's core package.

Classes:
    Model: Base model holding the main structure after training
    Project: The main class that one can instantiate in order to
        start playing around with Falcondale's functionalities

        $ myproject = Project(dataset, target="target")
"""

import pickle
import warnings
import pandas as pd

from .data import Dataset
from .classifiers import qsvc, qnn
from .feature_selection import qfs, qfs_neal, qfs_sb, qfs_sim_qaoa
from .clustering import prob_q_clustering
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


class Model:
    """
    Model structure is the object class of models
    trained with Falcondale Project's evaluate function.

    Holds information regarding the model but also contains
    required preprocessing steps (normalization, dimensionality reduction,...)
    as well as the metrics obtained against testing data split while training.
    """

    def __init__(self, dataset: Dataset, m_type: str, model, classification_report, scores: dict):
        """
        Inits the model structure with provided information.

        Args:
            dataset (Dataset): A Falcondale Dataset data structure.
            m_type (str): Model type name associated with the training phase.
                Should coincide with the options available in the evaluate method within Project.
            model (object): A generic object with at least a predict function enabled.
            classification_report (object): A sklearn.classification_report type of object
            scores (dict): A dictionary containing the metrics for the model
        """
        self._data = dataset
        self._type = m_type
        self._model = model
        self._creport = classification_report
        self._scores = scores

    def predict(self, data: pd.DataFrame) -> list[int]:
        """
        Predict function applied to the provided dataframe. It
        also takes care of shaping it so that it will fit
        in the scheme that the model expects.

        Args:
            data (pd.DataFrame): Pandas DataFrame

        Returns:
            list[int]: Predicted class
        """
        features_df = self._data.transform(data)
        return self._model.predict(features_df)

    def predict_proba(self, data: pd.DataFrame) -> list[float]:
        """
        Probability of predicting the 1 class applied to the
        provided dataframe. It also takes care of shaping it so
        that it will fit in the scheme that the model expects.

        Args:
            data (pd.DataFrame): Pandas DataFrame

        Returns:
            list[float]: Probability of predicting the 1 class
        """
        features_df = self._data.transform(data)
        return self._model.predict_proba(features_df)[:, -1]

    def list_metrics(self) -> list[str]:
        """
        Returns the metric values stored provided by the training funnel.
        This metrics are meant to evaluate models accuracy from different perspectives
        and have a direct access to metrics contained within the classification report
        in an easier way.

        This metrics are collected during training time depending on each model but
        in a general case will contain:

        * Specificity or True Negative Rate
        * Recall, sensitivity or True Positive Rate
        * Precision or Positive Predictive Value
        * Accuracy
        * Balanced accuracy
        * AUC
        * F1 score

        Returns:
            list[str]: List of available metrics.

        Example:
            ```py
            model.list_metrics()
            ```
        """
        return list(self._scores.keys())

    def metric(self, metric_name: str) -> float:
        """
        Returns the metric values against the test dataset during
        model training.

        Args:
            metric_name (str): Metric name to obtain the value from

        Returns:
            float: Value of the metric

        Example:
            ```py
            model.metric('auc')
            ```
        """
        if metric_name in self._scores:
            return self._scores[metric_name]

        print("Requested metric not available")

    def print_report(self):
        """
        Prints the classification report of the model taken from
        sklearns classification report similar to:

                          precision    recall  f1-score   support

                       0       0.83      0.75      0.79        60
                       1       0.87      0.92      0.89       111

                accuracy                           0.86       171
               macro avg       0.85      0.83      0.84       171
            weighted avg       0.86      0.86      0.86       171
        """
        print(self._creport)


class Project:
    """
    Base class defining the set of steps to be performed for a given project.

    Wraps complex functionality calls and framework integrations so that it is
    simpler for the user to call specific training or data processing functions.
    """

    def __init__(self, input_data: pd.DataFrame, target: str):
        """
        Initializes the Project class to perform a evaluations on different QML methods.

        Falcondale is currently focused on supervised methods (even though it supports some
        unsupervised methods for data clustering), that is why a target variable needs to be
        provided.

        Args:
            input_data (pd.DataFrame): Pandas DataFrame
            target (str): Target variable to be selected from the
                input_data dataframe
        """
        self._data = Dataset(input_data, target)

    def preprocess(self, reduced_dimension: int = None, method: str = None):
        """
        Performs information preprocessing to cleans the dataset. This is a mandatory
        step in most cases as it helps the Falcondale framework to foresee some of the
        tasks needed moving forward in particular for when 'auto' option is selected.

        Mostly focused on data normalization, gap filling (missing values)
        and duplicate entry removal. It also allows for simple dimensionality
        reduction by indicating the reduced dimension size. This feature is
        mainly used when specific models are selected as large feature spaced may
        require more computational resources than the ones available on a common laptop.

        Args:
            reduced_dimension (int): (Optional) Reduces initial feature set to a lower
                dimensional space.
            method (str): (Optional) Currently only supports PCA in combination with
                reduced_dimension parameter.

        Example:
            ```py
            myproject.preprocess(reduced_dimension=3)
            ```
        """
        self._data.preprocess(reduced_dimension=reduced_dimension, method=method)

    def profile_dataset(self):
        """
        Profiles the dataset in order to identify potential issues with it.

        It is basically a wrapper over YData's data profiler.
        """
        return self._data._profile()

    def show_features(self) -> pd.DataFrame:
        """
        Returns the features as they are internally used by the framework.

        Returns:
            pd.DataFrame: Pandas dataframe with the actual features being used within the Framework
        """
        return self._data.get_features()

    def feature_selection(self, max_cols: int, method: str = "sa", **kwargs) -> list[str]:
        """
        Performs the quantum feature selection to reduce the columns to be used selecting
        the obtained features as the ones to be used in following steps.

        Currently supports binary classification methods:

        * **sa**: For Simulated Annealing
        * **sb**: For Simulated Bifurcation
        * **qa**: For Quantum Annealing (also if token is provided)
        * **qaoa**: For Quantum Approximate Optimization Algorithm

        Examples:
            Can be called from Project instance:

            ```py

            features = myproject.feature_selection(max_cols, method)
            ```

        Args:
            max_cols (int): Maximum number of features to be selected
            method (str): (default qa) Method to be used when trying to find the
        optimal set of features

        Returns:
            list[str]: The list of selected features.

            This are also the features that will be included in the project as the current
            selection of features for following steps within the project.

        """
        method = method.lower()

        if max_cols > len(self._data._columns):
            print("Your previous selection is in force, preprocess the dataset to start over.")
            return None

        if method == "qaoa":
            if len(self._data.columns()) > 10:
                print(
                    f"{len(self._data.columns())} columns may require too much memory and can cause the kernel to fail."
                )

            feature_cols = qfs_sim_qaoa(max_cols=max_cols, input_ds=self._data)
            self._data.set_features(feature_cols)

            return feature_cols
        elif method == "sb":
            # Locally simulate using simulated bifurcation
            feature_cols = qfs_sb(max_cols=max_cols, input_ds=self._data)
            self._data.set_features(feature_cols)

            return feature_cols
        else:
            if "token" in kwargs:
                token = kwargs["token"]

                feature_cols = qfs(token=token, max_cols=max_cols, input_ds=self._data)
                self._data.set_features(feature_cols)

                return feature_cols
            else:
                feature_cols = qfs_neal(max_cols=max_cols, input_ds=self._data)
                self._data.set_features(feature_cols)

                return feature_cols

    def list_options(self):
        """
        Lists what can be done with Falcondale. A simple hint on the available options.
        """
        print("Welcome to Falcondale SDK! you will be able to:")
        print(" - Perform Quantum Feature Selection using Quantum Annealing or QAOA")
        print(" - Train quantum-enhanced models such as QSVC or QNN")
        print(" - Perform Quantum Clustering by pure quantum and quantum-inspired techniques")

    def evaluate(self, model: str, test_size: float = 0.3, **kwargs) -> Model:
        """
        Given a set of methods implements their training and evaluates
        performance for the initially given dataset.

        Currently supports binary classification methods:

        * **qsvc**: For Quantum Support Vector Classifier
        * **qnn**: For Quantum Neural Network

        Args:
            model (str): Model name to be trained
            test_size (float): (Optional) Test split size ratio.30% default

        Returns:
            Model: A Falcondale Model instance holding the trained model
                and some additional information

        Examples:
            Can be called from Project instance:

            ```py

            model = myproject.evaluate('<model_type>')
            ```

        """
        model = model.lower()

        verbose = False
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]

        backend = "qiskit"
        if "backend" in kwargs:
            backend = kwargs["backend"]

        if model == "qsvc":
            qsvc_model, report, metrics = qsvc(
                dataset=self._data, test_size=test_size, backend=backend, verbose=verbose
            )
            fmodel = Model(self._data, model, qsvc_model, report, metrics)
            return fmodel

        if model == "qnn":
            if "layers" in kwargs:
                layers = kwargs["layers"]

                qnn_model, report, metrics = qnn(
                    dataset=self._data, test_size=test_size, layers=layers, verbose=verbose
                )
            else:
                qnn_model, report, metrics = qnn(dataset=self._data, test_size=test_size, verbose=verbose)

            fmodel = Model(self._data, model, qnn_model, report, metrics)
            return fmodel

        return NotImplemented

    def cluster(self, ctype: str, **kwargs) -> tuple[list[int], list[float]]:
        """
        Given a type of clustering it is implemented using the
        information of the inner Dataset. Essentially takes the features
        to produce an estimate on the different segments it could be composed by.

        Currently supports:

        * **pqc**: For Probabilistic Quantum Clustering

        Examples:
            Can be called from Project instance:

            ```py

            label, label_prob = myproject.cluster(ctype='<model_type>')
            ```

        Args:
            ctype (str): Type of clustering to be performed

        Returns:
            (list[int], list[float]): A tuple containing the assigned cluster label
                and its likelihood or label probability.
        """
        ctype = ctype.lower()

        if ctype == "pqc":
            sigma = None
            if "sigma" in kwargs:
                sigma = kwargs["sigma"]

            return prob_q_clustering(input_ds=self._data, forced_sigma=sigma)

        return NotImplemented

    def save_model(self, model: Model, name: str):
        """
        Saves the model under the filename provided.

        Args:
            model (Model): A Falcondale model type of object
            name (str): File name
        """
        with open(name, "wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, name: str) -> Model:
        """
        Loads a Falcondale type of model from the
        provided file name.

        Args:
            name (str) : File name

        Returns:
            Model: Falcondale Model type
        """
        with open(name, "rb") as handle:
            model = pickle.load(handle)

        return model
