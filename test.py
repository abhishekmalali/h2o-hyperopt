from gbmoptimizer import *


def test_gbm():
    newgbm = GBMOptimizer(metric='auc')
    # Printing all the default parameters
    newgbm.default_parameters_opt()
    # Second test
    newgbm = GBMOptimizer(metric='auc')
    newgbm.select_optimization_parameters({'ntrees': ('choice', [10, 20, 30]),
                                           'max_depth': ('randint', (2, 50)),
                                           'learn_rate': 'Default'})
    newgbm.add_optimization_parameters({'col_sample_rate':
                                        ('uniform', (0.5, 0.8))})
    newgbm.rem_optimization_parameters(['col_sample_rate'])

if __name__ == '__main__':
    test_gbm()
