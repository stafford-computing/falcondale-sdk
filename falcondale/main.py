import time
import json
import pandas as pd
from io import BufferedReader, BytesIO
from typing import Optional

import requests

class Falcondale:

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret_key: Optional[str] = None,
                 api_server_url: str = "https://api.falcomdale.io"):

        self._api_server_url = api_server_url
        self._trained_file = None

        #TODO: Taken from API secret auth
        self._user_id = "test_user"

    @property
    def user_id(self):
        return self._user_id

    @user_id.setter
    def user_id(self, user_id):
        self._user_id = user_id

    def upload_dataset(self, local_file: str, is_training:bool = True):

        if (self._user_id is None):
            print("User not defined. Call 'set_user' first, with your ID")

            return False
        
        if (not self._check_limits(local_file)):
            print("Dataset is too big. Should be under 500 observations and 300 features")
            return False

        url = f"{self._api_server_url}/upload/{'training' if is_training else 'predict'}/{self._user_id}"

        f = open(local_file, 'rb')
        files = {"file": (f"{local_file}", f, "multipart/form-data")}

        r = requests.post(url=url, files=files)

        if r.status_code == 200:
            self._response= r.json()
            print(r.json())

            return True
        else:
            print("Something went wrong.")

            return False
        
    def _check_limits(self, local_file: str):

        df = pd.read_csv(local_file)

        if len(df) > 500 | len(df.columns) > 300:
            return False 

        return True

    def train(self,
              model_type: str,
              target_variable: str,
              csv_data_filename = str,
              model_backend: str = "qiskit",
              feature_selection_type: str = "",
              feature_selection_backend: str = "",
              is_async: bool = False) -> str:

        self._model_type = model_type
        self._model_backend = model_backend
        self._feature_selection_type = feature_selection_type
        self._feature_selection_backend = feature_selection_backend
        self._target_variable = target_variable
        self._csv_data_filename = csv_data_filename

        if (self._user_id is None):
            print("User not defined. Call 'set_user' first, with your ID")

            return ""

        url = f"{self._api_server_url}/train/{self._user_id}"

        data = {
            "model_backend" : self._model_backend,
            "model_type" : self._model_type,
            "feature_selection_type" : self._feature_selection_type,
            "feature_selection_backend" : self._feature_selection_backend,
            "target" : self._target_variable,
            "filename" : self._csv_data_filename
        }

        r = requests.post(url=url, json=data)

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
        else:
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
                csv_data_filename: str,
                is_async: bool = False):
        
        if (self._user_id is None):
            print("User not defined. Call 'set_user' first, with your ID")

            return ""


        url = f"{self._api_server_url}/predict/{self._user_id}"
        
        data = {
            "filename" : csv_data_filename,
            "model_name" : model_name
        }

        r = requests.post(url=url, json=data)

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
        else:
            print("Something went wrong.")

            return ""
    
    def get_current_workflow_id(self):

        return self._response

    def feature_selection(self,
                selection_type: str,
                target: str,
                csv_data_filename: str,
                token: str = ""):

        url = f"{self._api_server_url}/feature-selection"

        data = {
            "filename" : csv_data_filename,
            "feature_selection_type" : selection_type,
            "token" : token,
            "target" : target
        }

        r = requests.post(url=url, json=data)

        if r.status_code == 200:
            self._response= r.json()

            return True
        else:
            print("Something went wrong.")

            return False

    def status(self, training_id):

        url = f"{self._api_server_url}/check-status"
        url += f"/{training_id}"

        r = requests.get(url=url)

        if r.status_code == 200:
            return r.json()
        else:
            print("Something went wrong.")
            return {}

    def get_training_result(self, training_id) -> str:

        url = f"{self._api_server_url}/result"
        url += f"/{training_id}"

        r = requests.get(url=url)

        if r.status_code == 200:
            self._response = r.json()
            return self._response

        else:
            print("Something went wrong.")
            return ""
