import json
from io import BufferedReader, BytesIO
from typing import Optional

import requests


class Falcodale:

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret_key: Optional[str] = None,
                 api_server_url: str = "https://api.falcodale.io"):

        self._api_server_url = api_server_url
        self._trained_file = None

    def train(self,
              model_type: str,
              model_backend: str,
              feature_selection_type: str,
              feature_selection_backend: str,
              target_variable: str,
              csv_data_filename: str,
              csv_data: BufferedReader) -> bool:

        self._model_type = model_type
        self._model_backend = model_backend
        self._feature_selection_type = feature_selection_type
        self._feature_selection_backend = feature_selection_backend
        self._target_variable = target_variable
        self._csv_data = csv_data

        url = f"{self._api_server_url}/train"
        url += f"?model_backend={self._model_backend}"
        url += f"&model_type={self._model_type}"
        url += f"&feature_selection_type={self._feature_selection_type}"
        url += f"&feature_selection_backend={self._feature_selection_backend}"
        url += f"&target_variable={self._target_variable}"

        r = requests.post(
            url=url, files={'csv_data': (csv_data_filename, self._csv_data)})

        if r.status_code == 200:
            self._score = r.json()[0]
            self._trained_file = r.json()[1]
            self._model_uuid = self._trained_file.split("_")[-1]

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

    def predict(self,
                model_name: str,
                csv_data_filename: str,
                csv_data: BufferedReader):

        if self._trained_file is None:
            print("Please train/load the model first")
            return


        url = f"{self._api_server_url}/predict"
        url += f"?model_name={model_name}"

        r = requests.post(
            url=url, files={'csv_data': (csv_data_filename, csv_data)})

        return r

    def load(self,
             model_uuid: str):
        raise NotImplementedError()
