from util import *
from modeloptimizer import *
from h2o.h2o import H2ODeepLearningEstimator

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
        self.model = H2ODeepLearningEstimator()
        self._hp_model_params = None
        self.trials = None
        self.best_model = None

    def _gen_score(self, params, model, metric):
        # Checking if the user decided to use cross-validation
        if 'nfolds' in params.keys():
            """
            # Need to check on cross_validation_metrics_summary() function

            cross_val_data = model.cross_validation_metrics_summary().\
                            as_data_frame()
            cross_val_data = cross_val_data.set_index('')
            cv_val = float(cross_val_data.loc[metric]['mean'])
            valid_val = gen_metric(model.model_performance(self.validFr),
                                   metric)
            score = (cv_val + valid_val)/2
            """
            score = gen_metric(model.model_performance(self.validFr), metric)
        else:
            score = gen_metric(model.model_performance(self.validFr), metric)
        if metric == 'auc':
            score = -score
        return score

    def objective_auto(self, params):
        metric = self._hp_model_params['metric']
        model = H2ODeepLearningEstimator()
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
