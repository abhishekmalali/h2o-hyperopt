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

    def testSelOptParameters(self):
        model = h2ohyperopt.ModelOptimizer(metric='auc')
        model.select_optimization_parameters(params=
                                        {'ntrees': ('choice', [10, 20, 30])})
        assert set(model.model_params.keys()) == set(['metric', 'ntrees'])

    def testAddRemOptParameters(self):
        model = h2ohyperopt.ModelOptimizer(metric='auc')
        model.select_optimization_parameters(params="Default")
        model.add_optimization_parameters({'col_sample_rate':
                                           ('uniform', (0.5, 0.8)),
                                           'ntrees': ('choice', [10, 20, 30])})
        assert set(model.model_params.keys()) == set(['metric', 'ntrees',
                                                      'col_sample_rate'])

        # Removing a certain parameter
        model.rem_optimization_parameters('col_sample_rate')
        assert set(model.model_params.keys()) == set(['metric', 'ntrees'])
