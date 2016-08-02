import h2o
from unittest import TestCase
import h2ohyperopt
from data_gen import *
h2o.init()


class TestGLMOpt(TestCase):
    def testGLM(self):
        model = h2ohyperopt.GLMOptimizer(metric='auc')
        model.select_optimization_parameters("Default")
        trainFr, testFr, validFr, predictors, response = data_gen()
        model.start_optimization(num_evals=10, trainingFr=trainFr,
                                 validationFr=validFr, response=response,
                                 predictors=predictors)
