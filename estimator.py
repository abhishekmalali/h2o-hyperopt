from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt
import h2o

#custom imports
from util import *


def_params_GBM =   {'learn_rate':hp.uniform('learn_rate', 0.01, 0.2),
                    'max_depth':randint('max_depth', 2, 20),
                    'ntrees': 200,
                    'col_sample_rate':hp.uniform('col_sample_rate', 0.5, 0.8),
                    'stopping_rounds': 10,
                    'sample_rate': hp.uniform('sample_rate', 0.8, 1.0),
                    'nfolds': 5}


def_params_DLE =  {'activation': hp.choice('activation',["Tanh", "TanhWithDropout",\
            "Rectifier", "RectifierWithDropout","Maxout","MaxoutWithDropout"]),
            'hidden': hp.choice('hidden',[50, 100, 150, 200]),
            'adaptive_rate': True,
            'rho': hp.uniform('rho', 0.5, 0.8)}



params_map = {'GBM': def_params_GBM,
                'DLE': def_params_DLE}


def updateMetric(params, metric):
    for i in range(len(params)):
        params[i]['metric'] = metric
    return params

def ModelEstimator(models='All', metric='auc', params='Default', optimize='Default'):
    """
    Models - models to be evaluated
    metric - loss metric
    params - 'Default' or give alternate - nested dictionary keyed in by model names
    optimize - 'Default' optimizes all parameters or give alternate parameters to optimize
    If you give params then no need to use optimize. Use optimize only when using the default params.
    Pass a dictionary with value as a list to the optimize keyword argument with parameters 
    """

    masterModelList = ['All', 'GBM', 'DLE']
    #Checking if the models are in the list of models we define
    assert(mod in masterModelList for mod in models), "Models not correctly defined"
    

    #In case all is passed into the model list with other manually chosen models
    if type(models) == list and 'All' in models:
        models = 'All'


    if models == 'All':
        modParams = []
        for mod in masterModelList[1:]:
            if params == 'Default' and optimize == 'Default': 
                modParams.append({'type': mod, 'params':params_map[mod]})
            elif params != 'Default':
                modParams.append({'type':mod, 'params':params[mod]})
        
    if type(models) == list:
        for mod in models:
            if params == 'Default' and optimize =='Default':
                modParams.append({'type': mod, 'params':params_map[mod]})
            elif params != 'Default':
                modParams.append({'type':mod, 'params':params[mod]})



    modParams = updateMetric(modParams, metric)


    return modParams





def objective(params):
    if params['type'] == 'GBM':
        model = h2o.H2OGradientBoostingEstimator()
    elif params['type'] == 'DLE':
        model = h2o.H2ODeepLearningEstimator()
