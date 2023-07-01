import time
import requests
import logging
import pandas as pd
from typing import Optional
from pathlib import Path
from interruptingcow import timeout

# Global configs
TIMEOUT_LIMIT = 180
MAX_ROWS = 500
MAX_COLS = 300


class Falcondale:
    """
    Main Falcondale class
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret_key: Optional[str] = None,
                 api_server_url: str = "https://api-falcondale.bsmk.xyz"):

        # API connection setup
        self._api_key = api_key
        self._api_secret = api_secret_key
        self._api_server_url = api_server_url

        # Set defaults
        self._response = None
        self._trained_file = None

        #TODO: Taken from API secret auth
        self._user_id = None

    @property
    def user_id(self):
        """
        User id property
        """
        return self._user_id

    @user_id.setter
    def user_id(self, user_id):
        """
        User id setter
        """
        self._user_id = user_id

    def upload_dataset(self,
                       local_file,
                       is_training:bool = True,
                       dataset_name:str = None):
        """
        Obtains the information needed to upload the dataset to User's storage space
        """

        if not self._user_id:
            logging.error("User not defined. Assign your user_id first: model.user_id = 'user_id'")
            return

        if not self._check_limits(local_file):
            logging.error("Dataset is too big. Should be under 500 observations and 300 features")
            return

        # Select the endpoint
        endpoint = 'training' if is_training else 'predict'
        url = f"{self._api_server_url}/upload/{endpoint}/{self._user_id}"

        req = None
        if isinstance(local_file, str):
            with open(local_file, 'rb') as file:
                if dataset_name:
                    file_name = dataset_name
                else:
                    file_name = Path(local_file).name
                files = {"file": (f"{file_name}", file, "multipart/form-data")}

                req = requests.post(
                    url=url,
                    files=files,
                    timeout=TIMEOUT_LIMIT # Timeout after 1.5 min
                )

        elif isinstance(local_file, pd.DataFrame) and dataset_name:
            csv_str = local_file.to_csv(index=False)
            files = {"file": (f"{dataset_name}", str.encode(csv_str), "multipart/form-data")}

            req = requests.post(
                url=url,
                files=files,
                timeout=TIMEOUT_LIMIT # Timeout after 1.5 min
            )

        else:
            logging.error("Either a local file is given or a Pandas dataframe and a datatset name associated with it.")
            return

        if req and req.status_code == 200:
            self._response= req.json()
            print("Data has beed correctly uploaded!")
            return

        #Else
        logging.error("Something went wrong.")

    def _check_limits(self, local_file):
        """
        Checks if the limits are met
        """
        # Evaluate the input
        if isinstance(local_file, str):
            local_df = pd.read_csv(local_file)
        else:
            local_df = local_file

        # Condition
        if len(local_df) > MAX_ROWS | len(local_df.columns) > MAX_COLS:
            return False

        return True

    @timeout(TIMEOUT_LIMIT)
    def train(self,
              model_type: str,
              target_variable: str,
              dataset_name = str,
              model_backend: str = "qiskit",
              qnn_layers: int = 3,
              feature_selection_type: str = "",
              feature_selection_token: str = "",
              validation_size: float = 0.4,
              is_async: bool = False) -> str:
        """
        Core functionality providing the information needed
        to train a model of the supported types.
        """

        self._model_type = model_type
        self._model_backend = model_backend
        self._feature_selection_type = feature_selection_type
        self._feature_selection_token = feature_selection_token
        self._target_variable = target_variable
        self._csv_data_filename = dataset_name
        self._validation_size = validation_size
        self._qnn_layers = qnn_layers

        if not self._user_id:
            logging.error("User not defined. Call 'set_user' first, with your ID")
            return

        if model_type == "QNN" and qnn_layers > 10:
            logging.info(f"{qnn_layers} layers could be too much. Pick a number below 10.")
            return

        url = f"{self._api_server_url}/train/{self._user_id}"

        data = {
            "model_backend" : self._model_backend,
            "model_type" : self._model_type,
            "feature_selection_type" : self._feature_selection_type,
            "feature_selection_token" : self._feature_selection_token,
            "target" : self._target_variable,
            "filename" : self._csv_data_filename,
            "validation_size": self._validation_size,
            "qnn_layers" : self._qnn_layers
        }

        response = requests.post(
            url=url,
            json=data,
            timeout=TIMEOUT_LIMIT
        )

        if response.status_code == 200:
            training_id = response.json()

            if not is_async:
                while True:
                    if self.status(training_id) != 'COMPLETED':
                        # print ("not ready, waiting 5 seconds")
                        time.sleep(5)
                    else:
                        return self.get_training_result(training_id)

            return training_id

        #Else
        logging.error("Something went wrong.")

    @timeout(TIMEOUT_LIMIT, TimeoutError)
    def predict(self,
                model_name: str,
                dataset_name: str,
                is_async: bool = False):

        if not self._user_id:
            logging.error("User not defined. Call 'set_user' first, with your ID")
            return

        url = f"{self._api_server_url}/predict/{self._user_id}"

        data = {
            "filename" : dataset_name,
            "model_name" : model_name
        }

        response = requests.post(
            url=url,
            json=data,
            timeout=TIMEOUT_LIMIT
        )

        if response.status_code == 200:
            prediction_id = response.json()

            if not is_async:
                while True:
                    if self.status(prediction_id) != 'COMPLETED':
                        # print ("not ready, waiting 5 seconds")
                        time.sleep(5)
                    else:
                        return self.get_training_result(prediction_id)

            return prediction_id

        #Else
        logging.error("Something went wrong.")

    def get_current_workflow_id(self):
        """
        Returns workflow ID
        """
        return self._response

    def feature_selection(self,
                selection_type: str,
                target: str,
                dataset_name: str,
                token: str = ""):

        url = f"{self._api_server_url}/feature-selection/{self.user_id}"

        data = {
            "filename" : dataset_name,
            "feature_selection_type" : selection_type,
            "token" : token,
            "target" : target
        }

        respose = requests.post(
            url=url,
            json=data,
            timeout=TIMEOUT_LIMIT
        )

        if respose.status_code == 200:
            self._response= respose.json()

            return True

        #Else
        logging.error("Something went wrong.")

    def status(self, training_id: str = None):
        """
        Checks the status of the job
        """

        url = f"{self._api_server_url}/check-status"
        if training_id:
            url += f"/{training_id}"
        else:
            url += f"/{self._response}"

        respose = requests.get(
            url=url,
            timeout=TIMEOUT_LIMIT
        )

        if respose.status_code == 200:
            return respose.json()

        #Else
        logging.error("Something went wrong.")

    def get_training_result(self, training_id: str = None) -> str:

        url = f"{self._api_server_url}/result"
        if training_id:
            url += f"/{training_id}"
        else:
            url += f"/{self._response}"

        respose = requests.get(
            url=url,
            timeout=TIMEOUT_LIMIT
        )

        if respose.status_code == 200:
            self._response = respose.json()
            return self._response

        #Else
        logging.error("Something went wrong.")
