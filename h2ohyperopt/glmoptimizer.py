from util import *
from modeloptimizer import *
from h2o.h2o import H2OGeneralizedLinearEstimator


class GLMOptimizer(ModelOptimizer):

    def __init__(self, metric=None, problemType='Classification'):
        """
        Initializing GBMOptimizer class.

        Input
        ---------------------
        metric: Metric used by H2O to evaluate models.
        problemType: Choose one from ['Regression', 'Classification', 'MultiClass']
        """
        # Initializing the GLMOptimizer
        # Setting the default search parameters
        self.optimized = False
        self.def_params = {'lambda_search': ('choice', [True, False]),
                           'alpha': ('uniform', (0, 1)),
                           'nlambdas': ('randint', (5, 20)),
                           'nfolds': 5,
                           'metric': metric}
        self.model_params = None
        self.model = H2OGeneralizedLinearEstimator()
        self._hp_model_params = None
        self.trials = None
        self.best_model = None
        self.family = self._problemType(problemType)

    def _problemType(self, prString):
        """Internal function to determine the family type argument for GLM's"""
        if prString == 'Classification':
            return "binomial"
        elif prString == 'Regression':
            return "gaussian"
        elif prString == 'MultiClass':
            return "multinomial"
        else:
            raise ValueError, "problemType not defined correctly"

    def _gen_score(self, params, model, metric):
        """ Custom scoring function for the GLMOptimizer. """
        # Checking if the user decided to use cross-validation
        if 'nfolds' in params.keys():
            # TODO: Check on compatibility of cross_validation_metrics_summary() with new versions of H2O
            score = gen_metric(model.model_performance(self.validFr), metric)
        else:
            score = gen_metric(model.model_performance(self.validFr), metric)
        if metric == 'auc':
            score = -score
        return score

    def objective_auto(self, params):
        """ Internal objective function for the GLMOptimizer class. """
        metric = self._hp_model_params['metric']
        model = H2OGeneralizedLinearEstimator(family=self.family)
        # Setting model parameters in order to begin training
        model = update_model_parameters_GLM(model, params)
        # Training the model
        model.train(x=self.predictors,
                    y=self.response,
                    training_frame=self.trainFr,
                    early_stopping_rounds=10)
        score = self._gen_score(params, model, metric)
        return {'loss': score, 'status': STATUS_OK, 'model': model,
                'params': params}
