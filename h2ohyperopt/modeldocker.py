from util import *
from dleoptimizer import *
from gbmoptimizer import *
from modeloptimizer import *


class ModelDocker(ModelOptimizer):
    def __init__(self, modelList, metric):
        """
        Initializing the ModelDocker class.

        Input
        -------------------
        modelList: List of ModelOptimizers
        metric: Evalution metric to be used by H2O
        """
        self.modelList = modelList
        self.optimized = False
        self._create_hyperopt_format()
        self.best_model = None
        self.metric = metric

    def _create_hyperopt_format(self):
        """ Internal function to convert dictionaries to hyperopt format. """
        docker_params = []
        for model in self.modelList:
            model_params = {}
            model_params['model'] = model.model
            model_params['params'] = model._hp_model_params
            if model.family is not None:
                model_params['params']['family'] = model.family
            docker_params.append(model_params)

        self.hp_docker_params = hp.choice('classifier', docker_params)

    def _gen_score(self, params, model, metric):
        """ Custom scoring function for the ModelDocker. """
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
        """ Internal objective function for the ModelDocker class. """
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
        """
        Function to start training models and optimize over the search space.

        Input
        ---------------------
        num_evals: Number of objective function evaluations.
        trainingFr: H2OFrame with trainig data.
        validationFr: H2OFrame with valdiation data.
        predictors: List of column names to be designated predictor columns.
        response: String indicating the response variable in the trainingFr.
        """
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
        """
        Function to return parameters of best model

        Output
        -----------------
        Tuple of Best model type and dictionary of parameters of the model.
        """
        if self.best_model is not None:
            return (self.trials.best_trial['result']['params']['model'].__class__.__name__,
                    self.trials.best_trial['result']['params']['params'])
        else:
            raise ValueError, 'Model not yet optimized'


    def best_model_test_scores(self, testFr):
        """Function to generate scores on test data """
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

    def best_in_class_ensembles(self, numModels=1):
        """
        Function to pick best model from each ModelOptimizer class and
        ensemble these models.
        """
        scores = []
        model_type = []
        for i in range(len(self.trials.trials)):
            scores.append(self.trials.trials[i]['result']['loss'])
            model_type.append(type(self.trials.trials[i]['result']['model']))
        index = sorted(range(len(scores)), key=lambda k: scores[k])
        modelTypeMasterList = set(model_type)
        modelList = []
        idx = 0
        modCount = 0
        for modType in modelTypeMasterList:
            while modCount < numModels:
                while model_type[index[idx]] != modType and modCount < numModels:
                    idx += 1
                modelList.append(self.trials.trials[index[idx]]['result']['model'])
                modCount +=1
            modCount = 0
            idx = 0
        self.create_ensembler_model(modelList)

    def score_ensemble(self, data):
        dataEn = self.create_ensembler_data(self.ensemble_model_list, data, train=True)
        testScore = gen_metric(self.ensemble_model.
                               model_performance(dataEn),
                               self.metric)
        return testScore

    def _build_corr_dataset(self):
        # Building the master H2OFrame which is to be used for Correlation
        for i in range(len(self.trials.trials)):
            val = self.trials.trials[i]['result']['model'].predict(self.trainFr)['predict']
            val_name = 'predict'+str(i)
            val.columns = [val_name]
            if i == 0:
                predFrame = val
            else:
                predFrame = predFrame.cbind(val)
        predFrameCor = predFrame.cor(use='complete.obs')
        return predFrameCor.as_data_frame()

    def smart_ensembling(self):
        # Picking one model and picking the next lowest correlated component
        scores = []
        model_type = []
        for i in range(len(self.trials.trials)):
            scores.append(self.trials.trials[i]['result']['loss'])
            model_type.append(type(self.trials.trials[i]['result']['model']))
        index = sorted(range(len(scores)), key=lambda k: scores[k])
        modelTypeMasterList = set(model_type)
        modelList = []
        idxList = []
        idx = 0
        for modType in modelTypeMasterList:
            while model_type[index[idx]] != modType:
                idx += 1
            modelList.append(self.trials.trials[index[idx]]['result']['model'])
            idxList.append(index[idx])
            idx = 0
        predFrameCor = self._build_corr_dataset()
        for model_idx in idxList:
            new_mod_id = predFrameCor['predict'+str(model_idx)].idxmin()
            modelList.append(self.trials.trials[new_mod_id]['result']['model'])
        self.create_ensembler_model(modelList)
