# Example script for using the DLEOptimizer in h2ohyperopt

import h2o
import sys
sys.path.append('')
import h2ohyperopt

# Initializing H2O
h2o.init()


# Function for preprocessing the data
def data():
    """
    Function to process the example titanic dataset.
    Train-Valid-Test split is 60%, 20% and 20% respectively.
    Output
    ---------------------
    trainFr: Training H2OFrame.
    testFr: Test H2OFrame.
    validFr: Validation H2OFrame.
    predictors: List of predictor columns for the Training frame.
    response: String defining the response column for Training frame.
    """
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
    predictors = trainFr.names[:]
    # Removing the response column from the list of predictors
    predictors.remove('survived')
    response = 'survived'
    return trainFr, testFr, validFr, predictors, response


def model():
    # Initializing the model
    dleModel = h2ohyperopt.DLEOptimizer(metric='auc')
    # Selecting parameters to optimize on
    dleModel.select_optimization_parameters({'epsilon': 'Default',
                                             'adaptive_rate': True,
                                             'hidden': ('choice', [[10, 20], [30, 40]]),
                                             'nfolds': 7})
    # Setting a certain parameter to default ensures the parameter is searched
    # on default search space. Ex - 'epsilon'.
    # As with 'hidden' parameter we can decide the distribution as well as the
    # parameters.
    # Static parameters like nfolds etc. can also be user defined.

    # Using the data() function to get required data
    trainFr, testFr, validFr, predictors, response = data()

    # Starting the optimization of hyperparameters
    dleModel.start_optimization(num_evals=10, trainingFr=trainFr,
                                validationFr=validFr, response=response,
                                predictors=predictors)
    print "Best Model Scores"
    print dleModel.best_model_scores()
    print "Test frame score: ", dleModel.best_model_test_scores(testFr)
    print "Best Model Parameters"
    print "------------------------"
    print dleModel.best_model_parameters()

if __name__ == '__main__':

    model()
