from util import *
from modeloptimizer import *
from h2o import H2OGradientBoostingEstimator

class GBMOptimizer(ModelOptimizer):

    def __init__(self, metric=None):
        # Initializing the GBMOptimizer
        # Setting the default search parameters
        self.optimized = False
        self.def_params = {'learn_rate': ('uniform', (0.01, 0.2)),
                           'max_depth': ('randint', (2, 20)),
                           'ntrees': ('choice', [100, 200]),
                           'col_sample_rate': ('uniform', (0.5, 0.8)),
                           'stopping_rounds': 10,
                           'sample_rate': ('uniform', (0.8, 1.0)),
                           'nfolds': 5,
                           'metric': metric}
        self.model_params = None
        self.model = H2OGradientBoostingEstimator()
        self._hp_model_params = None
        self.trials = None
        self.best_model = None

    def objective_auto(self, params):
        metric = self._hp_model_params['metric']
        model = H2OGradientBoostingEstimator()
        if 'max_depth' in params.keys() and params['max_depth'] < 2:
            params['max_depth'] = 2
        # Setting model parameters in order to begin training
        model = update_model_parameters(model, params)
        # Training the model
        model.train(x=self.predictors,
                    y=self.response,
                    training_frame=self.trainFr,
                    early_stopping_rounds=5)
        score = self._gen_score(params, model, metric)
        return {'loss': score, 'status': STATUS_OK, 'model': model,
                'params': params}
