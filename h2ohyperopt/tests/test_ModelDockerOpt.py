import h2o
from unittest import TestCase
import h2ohyperopt
from data_gen import *
h2o.init()


class TestModelDockerOpt(TestCase):
    def testModelDocker(self):
        model_gbm = h2ohyperopt.GBMOptimizer(metric='auc')
        model_dle = h2ohyperopt.DLEOptimizer(metric='auc')
        model_glm = h2ohyperopt.GLMOptimizer(metric='auc')

        # Initializing as default
        model_gbm.select_optimization_parameters("Default")
        model_dle.select_optimization_parameters("Default")
        model_glm.select_optimization_parameters("Default")

        newdock = h2ohyperopt.ModelDocker([model_gbm, model_dle,
                                           model_glm], 'auc')

        trainFr, testFr, validFr, predictors, response = data_gen()

        newdock.start_optimization(num_evals=10, trainingFr=trainFr,
                                   validationFr=validFr, response=response,
                                   predictors=predictors)
        newdock.best_model_parameters()
        newdock.best_model_scores()
        newdock.best_in_class_ensembles(numModels=1)
        assert newdock.best_model_test_scores(testFr) > 0.5
