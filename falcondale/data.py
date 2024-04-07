"""
This module implements the data structure that will be used
along any steps of a Falcondale project.
"""

import json
from pandas import DataFrame, concat
from ydata_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class Dataset:
    """
    Dataset class gets the core information needed
    for Classical Data embedding.
    """

    def __init__(self, input_df: DataFrame, target: str = None):
        """
        This function initializes Falcondale's core dataset abstraction.

        It gets the dataset so that additional actions can be performed.

        Parameters:
        input_df (DataFrame): Input pandas dataframe
        target (str): Target column variable name
        """
        self._rawdataset = input_df
        self._columns = [col for col in input_df.columns if col != target]
        self.target = target

        # Init additional resources
        self._dim_reduce = None
        self._scaler = None
        self._metadata = None
        self._features = None
        self._df_profile = None

    def _profile(self):
        """
        Profiles dataset providing information about it

        Parameters:
        report_format (str): Defines the format of the report
        """

        profile = ProfileReport(self._rawdataset, minimal=True)
        self._df_profile = json.loads(profile.to_json())

        # Print
        print(f"Number of samples on the dataset: {self._df_profile['table']['n']}")
        print(f"Number of columns on the dataset: {self._df_profile['table']['n_var']}")
        if self.target:
            print(f"{self.target} has been selected as target variable for supervised learning tasks.")
        print(f"Ratio of samples with missing values: {self._df_profile['table']['p_cells_missing']}")

        if self._df_profile["table"]["n_var"] == self._df_profile["table"]["types"]["Numeric"]:
            print("All data is numeric therefore little preprocessing might need to be done.")
        else:
            print(
                "Seems like some features are not Numeric, make sure dataset counts with numeric values for categorical values as well."
            )

        if len(self._df_profile["alerts"]) > 0:
            print("Some relevant alerts have been found:")
            for alert in self._df_profile["alerts"]:
                print(f"\t * {alert}")

    def preprocess(self, reduced_dimension: int = None, method: str = "PCA"):
        """
        Preprocessing of the dataset according to defined criteria.

        Basically, inputs missing data and scales the numerical
        information representing the original dataset.
        """
        # Remove duplicates
        # if self._df_profile and self._df_profile['table']['duplicate_row_count'] > 0:
        #    self._rawdataset.drop_duplicates(inplace=True)

        self._features = self._rawdataset[self._columns].copy()

        # Simple zero input
        self._features.fillna(0)

        # Scale values for better embedding
        scaler = MinMaxScaler()
        self._scaler = scaler.fit(self._features.to_numpy())
        self._columns = self._features.columns
        scaled = self._scaler.transform(self._features.to_numpy())
        self._features = DataFrame(scaled, columns=self._columns)

        if reduced_dimension:
            if method == "LDA":
                # TODO pending
                # correlations = self._features.corrwith(self._rawdataset[self.target])
                # lda = LDA(n_components=reduced_dimension)

                raise NotImplementedError
            else:
                pca = PCA(n_components=reduced_dimension)
                reduced_features = pca.fit(self._features).transform(self._features)
                self._dim_reduce = pca
                self._columns = pca.get_feature_names_out()
                pca_cols = DataFrame(reduced_features, columns=self._columns)
                self._features = concat([self._features, pca_cols], axis=1)

    def transform(self, input_df: DataFrame):
        """
        Takes input dataframe and transforms it into the same shape
        as the original dataset was treated
        """
        # Remove duplicates
        output = input_df.copy()

        columns_names = output.columns

        # Scale values for better embedding
        scaled = self._scaler.transform(output.to_numpy())
        output = DataFrame(scaled, columns=columns_names)

        if self._dim_reduce:
            reduced_features = self._dim_reduce.transform(output)
            reduced_cols = self._dim_reduce.get_feature_names_out()
            pca_cols = DataFrame(reduced_features, columns=reduced_cols)
            output = concat([output, pca_cols], axis=1)

        return output[self._columns]

    def get_target(self) -> DataFrame:
        """
        Obtain data for the target column

        Returns:
        DataFrame: subset information from _features dataframe
        """
        return self._rawdataset[[self.target]]

    def get_features(self) -> DataFrame:
        """
        Obtain data for the feature columns

        Returns:
        DataFrame: subset information from _features dataframe
        """

        return self._features[self._columns]

    def set_features(self, feat_list: list) -> DataFrame:
        """
        Sets the subset of features
        """
        self._columns = feat_list

    def select(self, col_list: list) -> DataFrame:
        """
        Obtain data from the selected columns

        Returns:
        DataFrame: subset information from _features dataframe
        """
        if self.target in col_list:
            col_list.remove(self.target)
            return concat([self._features[col_list], self._rawdataset[self.target]], axis=1)

        return self._features[col_list]

    def columns(self) -> list:
        """
        Returns feature column names as a list.

        Returns:
        list: column names
        """
        if self._features is not None:
            return self._columns

        raise Exception("No features selected yet! Try calling preprocess() first.")

    def size(self) -> int:
        """
        Returns the size of the features to be encoded

        Returns:
        int: Size of the features
        """
        return self._features[self._columns].shape

    def train_test_split(self, test_size: int = 0.3, random_state: int = 12345):
        """
        Splits the inner DataFreame into train and tests blocks

        Returns:
        test_size(float): Ratio to split into
        random_state(int): Seed for the splitting
        """
        features = self._features[self._columns]
        target = self._rawdataset[[self.target]]

        return train_test_split(
            features,
            target.values.ravel(),
            test_size=test_size,
            random_state=random_state,
        )
