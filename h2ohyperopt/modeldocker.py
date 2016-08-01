from util import *
from dleoptimizer import *
from gbmoptimizer import *
from modeloptimizer import *


class ModelDocker(ModelOptimizer):
    def __init__(self, modelList, metric):
        self.modelList = modelList
        self.optimized = False
        self._create_hyperopt_format()
        self.best_model = None
        self.metric = metric

    def _create_hyperopt_format(self):
        docker_params = []
        for model in self.modelList:
            model_params = {}
            model_params['model'] = model.model
            model_params['params'] = model._hp_model_params
            docker_params.append(model_params)

        self.hp_docker_params = hp.choice('classifier', docker_params)

    def objective_auto(self, params):
        model = params['model']
        model_params = params['params']
        model = update_model_parameters(model, model_params)
        model.train(x=self.predictors,
                    y=self.response,
                    training_frame=self.trainFr,
                    early_stopping_rounds=5)
        score = self._gen_score(params, model, self.metric)
        return {'loss': score, 'status': STATUS_OK, 'model': model,
                'params': params}

    def start_optimization(self, num_evals=100, trainingFr=None,
                           validationFr=None, predictors=None,
                           response=None):
        self.trials = Trials()
        self.trainFr = trainingFr
        self.validFr = validationFr
        self.response = response
        self.predictors = predictors
        best_model = fmin(self.objective_auto, space=self.hp_docker_params,
                          algo=tpe.suggest, max_evals=num_evals,
                          trials=self.trials)
        self.best_model = self.trials.best_trial['result']['model']
        self.optimized = True

    def best_model_parameters(self):
        if self.best_model is not None:
            return (self.trials.best_trial['result']['params']['model'].__class__.__name__,
                    self.trials.best_trial['result']['params']['params'])
        else:
            raise ValueError, 'Model not yet optimized'


    def best_model_test_scores(self, testFr):
        testScore = gen_metric(self.best_model.
                               model_performance(testFr),
                               self.metric)
        return testScore

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
                                  self.metric)
            trainScore = gen_metric(self.best_model.
                                    model_performance(self.trainFr),
                                    self.metric)
            if return_value is False:
                print "The training loss metric" + "("+self.metric+") is :", trainScore
                print "The validation loss metric" + "("+self.metric+") is :", valScore
            else:
                return {"Training Score": trainScore,
                        "Validation Score": valScore}
