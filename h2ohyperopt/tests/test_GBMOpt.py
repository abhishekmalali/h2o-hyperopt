import h2o
from unittest import TestCase
import h2ohyperopt
from data_gen import *
h2o.init()



class TestGBMOpt(TestCase):
    def testGBM(self):
        self.model = h2ohyperopt.GBMOptimizer(metric='auc')
        self.model.select_optimization_parameters({'ntrees': 20,
                                              'col_sample_rate':
                                              ('uniform', (0.5, 0.8))})
        trainFr, testFr, validFr, predictors, response = data_gen()
        self.model.start_optimization(num_evals=10, trainingFr=trainFr,
                                 validationFr=validFr, response=response,
                                 predictors=predictors)
        best_model = self.model.return_best_model()
        best_model_score = self.model.best_model_scores(return_value=True)
        best_model_test_score = self.model.best_model_test_scores(testFr)
        assert best_model_test_score > 0.5
        best_model_parameters = self.model.best_model_parameters()
        # Testing ensembles
        self.model.best_model_ensemble()
        # Testing ensemble predictions
        ensemble_pred = self.model.predict_ensemble(testFr)
