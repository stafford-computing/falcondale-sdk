import json
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

    def upload_dataset(self, local_file: str):

        url = f"{self._api_server_url}/upload"

        f = open(local_file, 'rb')
        files = {"file": (local_file, f, "multipart/form-data")}

        r = requests.post(url=url, files=files)

        if r.status_code == 200:
            self._response= r.json()

            return True
        else:
            print("Something went wrong.")

            return False

    def train(self,
              model_type: str,
              target_variable: str,
              csv_data_filename = str,
              model_backend: str = "qiskit",
              feature_selection_type: str = "",
              feature_selection_backend: str = "") -> bool:

        self._model_type = model_type
        self._model_backend = model_backend
        self._feature_selection_type = feature_selection_type
        self._feature_selection_backend = feature_selection_backend
        self._target_variable = target_variable
        self._csv_data_filename = csv_data_filename

        url = f"{self._api_server_url}/train"

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
            self._response= r.json()

            return True
        else:
            print("Something went wrong.")

            return False

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

    def predict_batch(self,
                model_name: str,
                csv_data_filename: str):

        if self._trained_file is None:
            print("Please train/load the model first")
            return


        url = f"{self._api_server_url}/predict"
        
        data = {
            "filename" : csv_data_filename,
            "model_name" : model_name
        }

        r = requests.post(url=url, json=data)

        if r.status_code == 200:
            self._response = r.json()

            return True
        else:
            print("Something went wrong.")

            return False
    
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

    def status(self):

        url = f"{self._api_server_url}/check-status"
        url += f"/{self._response}"

        r = requests.get(url=url)

        if r.status_code == 200:
            return r.json()
        else:
            print("Something went wrong.")
            return {}

    def collect(self) -> str:

        url = f"{self._api_server_url}/collect"
        url += f"/{self._response}"

        r = requests.get(url=url)

        if r.status_code == 200:
            if "predict" in self._response:
                json_resp = r.json()
                return json_resp["labels"]
            elif "selection" in self._response:
                return r.json()
            else:
                report, model_file = r.json()
                self._trained_file = model_file

                return report
        else:
            print("Something went wrong.")
            return ""
