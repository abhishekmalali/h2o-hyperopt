from unittest import TestCase
import h2ohyperopt

class TestModelOptParameters(TestCase):
    def testDefParameters(self):
        model = h2ohyperopt.ModelOptimizer(metric='auc')
        model.default_parameters_opt(return_dict=False)
        params_list = model.default_parameters_opt(return_dict=True)
        assert params_list == model.def_params

    def testModelParameters(self):
        model = h2ohyperopt.ModelOptimizer(metric='auc')
        model.select_optimization_parameters(params="Default")
        model.model_parameters_opt(return_dict=False)
        params_list = model.model_parameters_opt(return_dict=True)
        assert params_list == model.model_params
