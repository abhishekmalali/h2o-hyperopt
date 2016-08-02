from util import *
from modeloptimizer import *
from h2o.h2o import H2OGeneralizedLinearEstimator


class GLMOptimizer(ModelOptimizer):

    def __init__(self, metric=None, problemType='Classification'):
        # problemType = ['Regression', 'Classification', 'MultiClass']
        # Initializing the GLMOptimizer
        # Setting the default search parameters
        self.optimized = False
        self.def_params = {'lambda_search': ('choice', [True, False]),
                           'nfolds': 5,
                           'metric': metric}
        self.model_params = None
        self.model = H2OGeneralizedLinearEstimator()
        self._hp_model_params = None
        self.trials = None
        self.best_model = None
        self.family = self._problemType(problemType)

    def _problemType(self, prString):
        if prString == 'Classification':
            return "binomial"
        elif prString == 'Regression':
            return "gaussian"
        elif prString == 'MultiClass':
            return "multinomial"
        else:
            raise ValueError, "problemType not defined correctly"

    def objective_auto(self, params):
        metric = self._hp_model_params['metric']
        model = H2OGeneralizedLinearEstimator(family=self.family)
        # Setting model parameters in order to begin training
        model = update_model_parameters(model, params)
        # Training the model
        model.train(x=self.predictors,
                    y=self.response,
                    training_frame=self.trainFr,
                    early_stopping_rounds=10)
        score = self._gen_score(params, model, metric)
        return {'loss': score, 'status': STATUS_OK, 'model': model,
                'params': params}
