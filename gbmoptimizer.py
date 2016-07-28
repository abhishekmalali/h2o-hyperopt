from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt
import h2o
import util
from util import *


class GBMOptimizer():

    def __init__(self, metric=None):
        # Initializing the GBMOptimizer
        # Setting the default search parameters
        self.hp_dict_constructed = False
        self.def_params_GBM = {'learn_rate': ('uniform', (0.01, 0.2)),
                               'max_depth': ('randint', (2, 20)),
                               'ntrees': ('choice', [100, 200]),
                               'col_sample_rate': ('uniform', (0.5, 0.8)),
                               'stopping_rounds': 10,
                               'sample_rate': ('uniform', (0.8, 1.0)),
                               'nfolds': 5,
                               'metric': metric}

        self.model = h2o.H2OGradientBoostingEstimator()

    def check_if_metric_set(self):
        """
        Function used to check whether the user has defined the error metric
        """
        assert self.def_params_GBM['metric'] != None,\
            "GBM requires the model to have a metric.\
            Use function set_metric()"

    def set_metric(self, metric):
        """
        Function used to explicitly set the model evaluation metric
        """
        self.def_params_GBM['metric'] = metric
        if self.model_params in locals():
            self.model_params['metric'] = metric

    def list_all_parameters(self):
        """
        Function to display all the parameters which can be optimized for GBM
        """
        print self.model._parms.keys()

    def default_parameters_opt(self):
        """
        Function to display the ranges of the default parameters over which
        optimization will take place.
        """
        for key in self.def_params_GBM.keys():
            if type(self.def_params_GBM[key]) == tuple:
                key_val = self.def_params_GBM[key]
                if key_val[0] == 'choice':
                    print key + ' : ' + key_val[0] + ' amongst the values  '\
                        + str(key_val[1])
                else:
                    print key + ' : ' + key_val[0] + ' distribution in range '\
                        + str(key_val[1])
            else:
                print "Static Parameter " + key + " : " + \
                        str(self.def_params_GBM[key])
        return

    def model_parameters_opt(self):
        """
        Function to display the ranges of the model parameters over which
        optimization will take place.
        """
        for key in self.model_params.keys():
            if type(self.model_params[key]) == tuple:
                key_val = self.model_params[key]
                if key_val[0] == 'choice':
                    print key + ' : ' + key_val[0] + ' amongst the values  '\
                        + str(key_val[1])
                else:
                    print key + ' : ' + key_val[0] + ' distribution in range '\
                        + str(key_val[1])
            else:
                print "Static Parameter " + key + " : " + \
                        str(self.model_params[key])
        return

    def select_optimization_parameters(self, params, use_default=True):
        """
        Function to select the default parameters
        This function can handle inputs as a list as well as a dictionary
        A dictionary input overwrites the default parameters

        For choosing the default values, initialize with "Default"
        """
        if params == "Default":
            self.model_params = self.def_params_GBM
            return
        self.model_params = {}
        if type(params) == list:
            for key in params:
                try:
                    self.model_params[key] = self.def_params_GBM[key]
                except:
                    raise ValueError, "Error in choosing parameters for GBM"
        elif type(params) == dict:
            for key in params.keys():
                try:
                    if params[key] == "Default":
                        self.model_params[key] = self.def_params_GBM[key]
                    else:
                        self.model_params[key] = params[key]
                except:
                    raise ValueError, "Error in choosing parameters for GBM"
        if 'metric' not in self.model_params.keys():
            self.model_params['metric'] = self.def_params_GBM['metric']
        self._create_hyperopt_format()
        return

    def add_optimization_parameters(self, params):
        for key in params.keys():
            if params[key] == "Default":
                self.model_params[key] = self.def_params_GBM[key]
            else:
                self.model_params[key] = params[key]
        self._create_hyperopt_format()
        return

    def rem_optimization_parameters(self, params):
        """
        Remove a variable from optimization variables dictionary
        Can take input as a list or a specific single variable
        """
        if type(params) == list:
            for p in params:
                try:
                    self.model_params.pop(p)
                except:
                    print p + " does not exist in model parameters"
                    pass
        else:
            try:
                self.model_params.pop(params)
            except:
                print params + " does not exist in model parameters"
                pass
        self._create_hyperopt_format()
        return
    # Internal function to build the model_params into the correct format
    # This is done to ensure format matches whats required for hyperopt

    def _select_dist_function(self, name, values):
        return getattr(util, values[0])(name, values[1])

    def _create_hyperopt_format(self):
        # Process self.model_params into hyperopt variables
        self._hp_model_params = {}
        for key in self.model_params.keys():
            if type(self.model_params[key]) == tuple:
                self._hp_model_params[key] = \
                    self._select_dist_function(key, self.model_params[key])
            else:
                self._hp_model_params[key] = self.model_params[key]
        return

    def objective_auto(self, params):
        metric = self._hp_model_params['metric']
        model = h2o.H2OGradientBoostingEstimator()
        if 'max_depth' in params.keys() and params['max_depth'] < 2:
            params['max_depth'] = 2
        # Setting model parameters in order to begin training
        base_attrs = dir(model)
        for key in params.keys():
            if key not in base_attrs and key != 'metric':
                raise ValueError("Wrong parameter %s passed to the model"
                                 % key)
            else:
                setattr(model, key, params[key])
        # Training the model
        model.train(x=self.predictors,
                    y=self.response,
                    training_frame=self.trainFr,
                    early_stopping_rounds=5)
        # Checking if the user decided to use cross-validation
        if 'nfolds' in params.keys():
            cross_val_data = model.cross_validation_metrics_summary().\
                            as_data_frame()
            cross_val_data = cross_val_data.set_index('')
            cv_val = cross_val_data.loc[metric]['mean']
            valid_val = gen_metric(model.model_performance(self.validFr),
                                   metric)
            score = (cv_val + valid_val)/2
        else:
            score = gen_metric(model.model_performance(self.validFr), metric)
        if metric == 'auc':
            score = -score
        print score
        return {'loss': score, 'status': STATUS_OK, 'model': model}

    # Function to start training and optimize over the surface
    def start_optimization(self, num_evals=100, trainingFr=None,
                           validationFr=None, predictors=None,
                           response=None):
        self.trials = Trials()
        self.trainFr = trainingFr
        self.validFr = validationFr
        self.response = response
        self.predictors = predictors
        print self._hp_model_params
        best_model = fmin(self.objective_auto, space=self._hp_model_params,
                          algo=tpe.suggest, max_evals=num_evals,
                          trials=self.trials)
