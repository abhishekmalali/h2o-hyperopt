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
    titanic_df = h2o.import_file(path="https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")

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
    gbmModel = h2ohyperopt.GBMOptimizer(metric='auc')
    # Selecting parameters to optimize on
    gbmModel.select_optimization_parameters({'col_sample_rate': 'Default',
                                             'ntrees': 200,
                                             'learn_rate': ('uniform',(0.05, 0.2)),
                                             'nfolds': 7})
    # Setting a certain parameter to default ensures the parameter is searched
    # on default search space.
    # As with learn_rate we can decide the distribution as well as the
    # parameters.
    # Static parameters like nfolds etc. can also be user defined.

    # Using the data() function to get required data
    trainFr, testFr, validFr, predictors, response = data()

    # Starting the optimization of hyperparameters
    gbmModel.start_optimization(num_evals=100, trainingFr=trainFr,
                                validationFr=validFr, response=response,
                                predictors=predictors)
    print "Best Model Scores"
    print gbmModel.best_model_scores()
    print "Test frame score: ", gbmModel.best_model_test_scores()
    print "Best Model Parameters"
    print "------------------------"
    print gbmModel.best_model_parameters()

if __name__ == '__main__':

    model()
