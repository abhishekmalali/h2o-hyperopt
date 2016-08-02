import h2o
from unittest import TestCase
import h2ohyperopt
from data_gen import *
h2o.init()



class TestGBMOpt(TestCase):
    def testGBM(self):
        model = h2ohyperopt.GBMOptimizer(metric='auc')
        model.select_optimization_parameters({'ntrees': 20,
                                              'col_sample_rate':
                                              ('uniform', (0.5, 0.8))})
        trainFr, testFr, validFr, predictors, response = data_gen()
        model.start_optimization(num_evals=10, trainingFr=trainFr,
                                 validationFr=validFr, response=response,
                                 predictors=predictors)
