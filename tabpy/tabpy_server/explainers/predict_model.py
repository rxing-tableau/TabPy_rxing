import numpy as np

class PrediectModel:
    def __init__(self, query_handler, model_name, uuid):
        self.query_handler = query_handler
        self.model_name = model_name
        self.query_uuid = uuid

    def predict(self, data):
        query_data = {'data_list': data}
        response = self.query_handler.python_service.ps.query(self.model_name, query_data, self.query_uuid)
        return np.array(response.response)
