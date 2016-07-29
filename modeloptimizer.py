from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt
import h2o
import util
from util import *


class ModelOptimizer():

    def __init__(self, metric=None):

        self.def_params = {}
        self.model_params = None
        self.model = None

    def check_if_metric_set(self):
        """
        Function used to check whether the user has defined the error metric
        """
        assert self.def_params['metric'] != None,\
        'GBM requires the model to have a metric. Use function set_metric()'
        print "Metric defined"

    def set_metric(self, metric):
        """
        Function used to explicitly set the model evaluation metric
        """
        self.def_params['metric'] = metric
        if self.model_params is not None:
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
        for key in self.def_params.keys():
            if type(self.def_params[key]) == tuple:
                key_val = self.def_params[key]
                if key_val[0] == 'choice':
                    print key + ' : ' + key_val[0] + ' amongst the values  '\
                        + str(key_val[1])
                else:
                    print key + ' : ' + key_val[0] + ' distribution in range '\
                        + str(key_val[1])
            else:
                print "Static Parameter " + key + " : " + \
                        str(self.def_params[key])
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
            self.model_params = self.def_params
            self._create_hyperopt_format()
            return
        self.model_params = {}
        if type(params) == list:
            for key in params:
                try:
                    self.model_params[key] = self.def_params[key]
                except:
                    raise ValueError, "Error in choosing parameters for GBM"
        elif type(params) == dict:
            for key in params.keys():
                try:
                    if params[key] == "Default":
                        self.model_params[key] = self.def_params[key]
                    else:
                        self.model_params[key] = params[key]
                except:
                    raise ValueError, "Error in choosing parameters for GBM"
        if 'metric' not in self.model_params.keys():
            self.model_params['metric'] = self.def_params['metric']
        self._create_hyperopt_format()
        return

    def add_optimization_parameters(self, params):
        for key in params.keys():
            if params[key] == "Default":
                self.model_params[key] = self.def_params[key]
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

    # Function to start training and optimize over the surface
    def start_optimization(self, num_evals=100, trainingFr=None,
                           validationFr=None, predictors=None,
                           response=None):
        self.trials = Trials()
        self.trainFr = trainingFr
        self.validFr = validationFr
        self.response = response
        self.predictors = predictors
        best_model = fmin(self.objective_auto, space=self._hp_model_params,
                          algo=tpe.suggest, max_evals=num_evals,
                          trials=self.trials)
        self.best_model = self.trials.best_trial['result']['model']
        self.optimized = True

    def return_best_model(self):
        """
        Return the best model
        """
        if self.optimized is True:
            return self.best_model
        else:
            raise ValueError, "Model not yet optimized."

    def best_model_scores(self, return_value=False):
        """
        Function that can be used to get scores of the best model on training
        and validation H2OFrames.
        Input
        ----------------
        return_value: If True function returns model
        """
        if self.best_model is None:
            raise ValueError, "Model not yet optimized"
        else:
            valScore = gen_metric(self.best_model.
                                  model_performance(self.validFr),
                                  self._hp_model_params['metric'])
            trainScore = gen_metric(self.best_model.
                                    model_performance(self.trainFr),
                                    self._hp_model_params['metric'])
            if return_value is False:
                print "The training loss metric" + "("+self._hp_model_params['metric']+") is :", trainScore
                print "The validation loss metric" + "("+self._hp_model_params['metric']+") is :", valScore
            else:
                return {"Training Score": trainScore,
                        "Validation Score": valScore}

    def best_model_test_scores(self, testFr):
        testScore = gen_metric(self.best_model.
                               model_performance(testFr),
                               self._hp_model_params['metric'])
        return testScore

    def _gen_score(self, params, model, metric):
        # Checking if the user decided to use cross-validation
        if 'nfolds' in params.keys():
            cross_val_data = model.cross_validation_metrics_summary().\
                            as_data_frame()
            cross_val_data = cross_val_data.set_index('')
            cv_val = float(cross_val_data.loc[metric]['mean'])
            valid_val = gen_metric(model.model_performance(self.validFr),
                                   metric)
            score = (cv_val + valid_val)/2
        else:
            score = gen_metric(model.model_performance(self.validFr), metric)
        if metric == 'auc':
            score = -score
        return score

    def best_model_parameters(self):
        # Return parameter of the best model discovered
        if self.best_model is not None:
            return self.trials.best_trial['result']['params']
        else:
            raise ValueError, 'Model not yet optimized'
