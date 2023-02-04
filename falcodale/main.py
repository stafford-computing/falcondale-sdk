from typing import Optional


class Falcodale:

    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret_key: Optional[str] = None,
                 api_server_url: Optional[str] = None):
        raise NotImplementedError()

    def model(self, type: str):
        raise NotImplementedError()
