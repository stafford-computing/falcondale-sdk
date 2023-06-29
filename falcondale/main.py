import time
import json
import pandas as pd
from io import BufferedReader, BytesIO
from typing import Optional
from pathlib import Path

from .utils import check_id

import requests

# Global configs
TIMEOUT_LIMIT = 90
MAX_ROWS = 500
MAX_COLS = 300


class Falcondale:
    """
    Main Falcondale class
    """
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret_key: Optional[str] = None,
                #  api_server_url: str = "https://api.falcomdale.io"):
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
            # TODO: Handle exception instead of print
            print("User not defined. Assign your user_id first: model.user_id = 'user_id'")
            return False

        if not check_id(self._user_id):
            # TODO: Handle exception instead of print
            print("Invalid user_id")
            return False

        if not self._check_limits(local_file):
            print("Dataset is too big. Should be under 500 observations and 300 features")
            return False

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
            print("Either a local file is given or a Pandas dataframe and a datatset name associated with it.")
            return False

        if req and req.status_code == 200:
            self._response= req.json()
            print("Data has beed correctly uploaded!")
            return True

        #Else
        print("Something went wrong.")
        return False

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

        if (self._user_id is None):
            print("User not defined. Call 'set_user' first, with your ID")

            return ""
        
        if not check_id(self._user_id):
            # TODO: Handle exception instead of print
            print("Invalid user_id")
            return False

        if model_type == "QNN" and qnn_layers > 10:
            print(f"{qnn_layers} layers could be too much. Pick a number below 10.")

            return ""

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

        r = requests.post(
            url=url,
            json=data,
            timeout=TIMEOUT_LIMIT
        )

        if r.status_code == 200:
            training_id = r.json()

            if (not is_async):
                while (True):
                    if (self.status(training_id) != 'COMPLETED'):
                        # print ("not ready, waiting 5 seconds")
                        time.sleep(5)
                    else:
                        return self.get_training_result(training_id)

            return training_id
        
        #Else
        print("Something went wrong.")
        return ""

    # def predict(self,
    #             **kwargs):

    #     if self._trained_file is None:
    #         print("Please train/load the model first")
    #         return

    #     binary_data = BytesIO(json.dumps(kwargs).encode())

    #     url = f"{self._api_server_url}/predict"
    #     url += f"?model_type={self._model_type}"
    #     url += f"&model_id={self._model_uuid}"

    #     r = requests.post(
    #         url=url, files={'csv_data': ("csv_data.csv", binary_data)})

    #     return r

    def predict(self,
                model_name: str,
                dataset_name: str,
                is_async: bool = False):
        
        if (self._user_id is None):
            print("User not defined. Call 'set_user' first, with your ID")

            return ""

        if not check_id(self._user_id):
            # TODO: Handle exception instead of print
            print("Invalid user_id")
            return False
        
        url = f"{self._api_server_url}/predict/{self._user_id}"
        
        data = {
            "filename" : dataset_name,
            "model_name" : model_name
        }

        r = requests.post(
            url=url,
            json=data,
            timeout=TIMEOUT_LIMIT
        )

        if r.status_code == 200:
            prediction_id = r.json()

            if (not is_async):
                while (True):
                    if (self.status(prediction_id) != 'COMPLETED'):
                        # print ("not ready, waiting 5 seconds")
                        time.sleep(5)
                    else:
                        return self.get_training_result(prediction_id)

            return prediction_id
        
        # Else
        print("Something went wrong.")
        return ""
    
    def get_current_workflow_id(self):

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

        r = requests.post(
            url=url,
            json=data,
            timeout=TIMEOUT_LIMIT
        )

        if r.status_code == 200:
            self._response= r.json()

            return True

        print("Something went wrong.")
        return False

    def status(self, training_id: str = None):

        url = f"{self._api_server_url}/check-status"
        if training_id:
            url += f"/{training_id}"
        else:
            url += f"/{self._response}"

        r = requests.get(
            url=url,
            timeout=TIMEOUT_LIMIT
        )

        if r.status_code == 200:
            return r.json()
        
        print("Something went wrong.")
        return {}

    def get_training_result(self, training_id: str = None) -> str:

        url = f"{self._api_server_url}/result"
        if training_id:
            url += f"/{training_id}"
        else:
            url += f"/{self._response}"

        r = requests.get(
            url=url,
            timeout=TIMEOUT_LIMIT
        )

        if r.status_code == 200:
            self._response = r.json()
            return self._response

        print("Something went wrong.")
        return None
