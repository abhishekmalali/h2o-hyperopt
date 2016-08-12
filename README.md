# H2ohyperopt
[![Build Status](https://travis-ci.org/abhishekmalali/h2o-hyperopt.svg?branch=master)](https://travis-ci.org/abhishekmalali/h2o-hyperopt)
[![PyPI version](https://badge.fury.io/py/h2ohyperopt.svg)](https://badge.fury.io/py/h2ohyperopt)
[![Coverage Status](https://coveralls.io/repos/github/abhishekmalali/h2o-hyperopt/badge.svg?branch=master)](https://coveralls.io/github/abhishekmalali/h2o-hyperopt?branch=master)

A package which is a convenience wrapper around Hyperopt for fast prototyping
with H2O models. H2ohyperopt lets the users utilize the power of Hyperopt without
learning to use it. Instead, just define optimizers with the parameters you want to
optimize over.

## Installation
```{python}
pip install h2ohyperopt
```
To install the package via github,
```{bash}
git clone https://github.com/abhishekmalali/h2o-hyperopt.git
cd h2o-hyperopt
python setup.py install
```

## Quick Start
### For a single model type
The following steps assume you have a clean dataset with a binary response column split into train and validation frames (you will need to provide the following variables with data: TrainFr, ValidFr, predictors, response).
```{python}
import h2ohyperopt
model = h2ohyperopt.GBMOptimizer(metric='auc')
# Selecting the optimization parameters
model.select_optimization_parameters({'col_sample_rate': 'Default',
                                      'ntrees': 200,
                                      'learn_rate': ('uniform',(0.05, 0.2)),
                                      'nfolds': 7})
```
The input format is (Type of distribution, (distribution parameters)). To optimize a model,
```{python}
# trainFr - Training frame
# validFr - Validation frame
# predictors - List of training variables
# response - String with response column name

model.start_optimization(num_evals=10, trainingFr=trainFr,
                         validationFr=validFr, response=response,
                         predictors=predictors)

#To get the best model
print model.best_model
```

### For a multiple model docker
A model docker is provided to run multiple models in the same optimization.
```{python}
model_gbm = h2ohyperopt.GBMOptimizer(metric='auc')
# Selecting the optimization parameters
model_gbm.select_optimization_parameters({'col_sample_rate': 'Default',
                                          'ntrees': 200,
                                          'learn_rate': ('uniform',(0.05, 0.2)),
                                          'nfolds': 7})

model_dle = h2ohyperopt.DLEOptimizer(metric='auc')
# Selecting parameters to optimize on
model_dle.select_optimization_parameters({'epsilon': 'Default',
                                        'adaptive_rate': True,
                                        'hidden': ('choice', [[10, 20], [30, 40]]),
                                        'nfolds':7})

docker = h2ohyperopt.ModelDocker([model_dle, model_gbm], 'auc')

docker.start_optimization(num_evals=10, trainingFr=trainFr,
                          validationFr=validFr, response=response,
                          predictors=predictors)
```
A model docker can build ensembles from the best of a Deep learning estimator and
a Gradient boosting estimator.
```{python}
# Building an ensemble using the two best models for each class.
docker.best_in_class_ensembles(numModels=2)
```

### Distributions provided in h2ohyperopt
1. uniform
2. randint
3. choice
4. normal
5. lognormal
6. qlognormal
7. quniform
8. qloguniform
9. loguniform
10. qnormal

## Metrics provided
1. auc
2. mse
3. logloss
4. r2

### TODO:
1. Smart ensembling of models.
2. Representative data based optimization to reduce time and resources spent.
