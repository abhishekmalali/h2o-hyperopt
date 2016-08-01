from gbmoptimizer import *
from dleoptimizer import *
from modeldocker import *
import h2o
h2o.init()


def data_gen():
    # 'path' can point to a local file, hdfs, s3, nfs, Hive, directories, etc.
    titanic_df = h2o.import_file(path="/Users/abhishek/Downloads/titanic.csv")

    # Basic preprocessing
    # columns_to_be_used - List of columns which are used in the training/test
    # data
    # columns_to_factorize - List of columns with categorical variables
    columns_to_be_used = ['pclass', 'age', 'sex', 'sibsp', 'parch', 'ticket',
                          'embarked', 'fare', 'survived']
    columns_to_factorize = ['pclass', 'sex', 'sibsp', 'embarked', 'survived']
    # Factorizing the columns in the columns_to_factorize list
    for col in columns_to_factorize:
        titanic_df[col] = titanic_df[col].asfactor()
    # Selecting only the columns we need
    titanic_frame = titanic_df[columns_to_be_used]
    trainFr, testFr, validFr = titanic_frame.split_frame([0.6, 0.2],
                                                         seed=1234)
    return trainFr, testFr, validFr


def test_docker():
    trainFr, testFr, validFr = data_gen()
    predictors = trainFr.names[:]
    # Removing the response column from the list of predictors
    predictors.remove('survived')
    response = 'survived'

    newdle = DLEOptimizer(metric='auc')
    newdle.select_optimization_parameters("Default")
    newgbm = GBMOptimizer(metric='auc')
    newgbm.select_optimization_parameters("Default")
    newdock = ModelDocker([newgbm, newdle], 'auc')

    newdock.start_optimization(num_evals=100, trainingFr=trainFr,
                               validationFr=validFr, response=response,
                               predictors=predictors)
    print newdock.best_model_parameters()
    print newdock.best_model_test_scores(testFr)
    print newdock.best_model_scores(return_value=True)
    newdock.best_model_ensemble(num_models=5)
    print newdock.ensemble_model
    #print newdock.save_best_model(path="../")


def test_dle():
    newdle = DLEOptimizer(metric='auc')
    newdle.select_optimization_parameters("Default")
    trainFr, testFr, validFr = data_gen()
    predictors = trainFr.names[:]
    # Removing the response column from the list of predictors
    predictors.remove('survived')
    response = 'survived'
    newdle.start_optimization(num_evals=10, trainingFr=trainFr,
                              validationFr=validFr, response=response,
                              predictors=predictors)
    print newdle.best_model_scores(return_value=True)
    print newdle.best_model_test_scores(testFr)
    print newdle.best_model_parameters()


def test_gbm():
    newgbm = GBMOptimizer(metric='auc')
    # Printing all the default parameters
    # Second test
    newgbm = GBMOptimizer()
    newgbm.select_optimization_parameters({'ntrees': ('choice', [10, 20, 30])})
    newgbm.set_metric('auc')
    newgbm.check_if_metric_set()
    newgbm.default_parameters_opt()
    newgbm.add_optimization_parameters({'col_sample_rate':
                                        ('uniform', (0.5, 0.8))})
    trainFr, testFr, validFr = data_gen()
    predictors = trainFr.names[:]
    # Removing the response column from the list of predictors
    predictors.remove('survived')
    response = 'survived'
    newgbm.start_optimization(num_evals=10, trainingFr=trainFr,
                              validationFr=validFr, response=response,
                              predictors=predictors)
    print newgbm.best_model_scores(return_value=True)
    print newgbm.best_model_test_scores(testFr)

if __name__ == '__main__':
    test_docker()
