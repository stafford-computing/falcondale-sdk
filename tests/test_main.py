import unittest
from sklearn import datasets

from falcondale import Project


class TestSum(unittest.TestCase):
    """
    Basic test suite
    """

    def setUp(self):
        # import some data to play with
        X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

        dataset = X.copy()
        dataset["target"] = y

        self.project = Project(dataset, target="target")

    def test_feature_selection(self):
        """
        Testing if the feature selection works
        and provides expected list.
        """
        self.project.preprocess()
        result = self.project.feature_selection(max_cols=10)

        self.assertEqual(len(result), 10, "List should contain 10 elements")


if __name__ == "__main__":
    unittest.main()
