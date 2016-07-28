from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt
import h2o
import util
from util import *
from modeloptimizer import *

class DLEOptimizer(ModelOptimizer):

    def __init__(self, metric=None):
        # Initialize the DLEOptimizer
        # Setting the default search parameters
        self.optimized = False
        self.def_params = {'epsilon': ('uniform', (0.1, 0.5)),
                           'adaptive_rate': True,
                           'hidden': ('choice', [100, 200, 300]),
                           'momentum_start': ('uniform', (0.4, 0.6)),
                           'nfolds': 5,
                           'metric': metric}
        self.model_params = None
        self.model = h2o.H2ODeepLearningEstimator()
        self._hp_model_params = None
        self.trials = None
        self.best_model = None

    def objective_auto(self, params):
        metric = self._hp_model_params['metric']
        model = h2o.H2ODeepLearningEstimator()
        # Setting model parameters in order to begin training
        model = update_model_parameters(model, params)
        # Training the model
        model.train(x=self.predictors,
                    y=self.response,
                    training_frame=self.trainFr,
                    early_stopping_rounds=10)
        score = self._gen_score(params, model, metric)
        return {'loss': score, 'status': STATUS_OK, 'model': model}
