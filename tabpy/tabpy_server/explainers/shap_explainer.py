import shap
import numpy as np

class ShapExplainer:
    # cached shapley values, {'token', {values}}
    shap_values = {}

    @classmethod
    def explain(cls, model, data, uuid):
        array = np.array(data['data_list'])
        explainer = shap.KernelExplainer(model, shap.sample(array))
        values = explainer.shap_values(array)
        #shap.summary_plot(values, array, plot_type="bar")
        ShapExplainer.shap_values[uuid] = values

    @classmethod
    def query_shap_values(cls, uuid):
        if uuid in ShapExplainer.shap_values.keys():
            return ShapExplainer.shap_values[uuid]
        return None
