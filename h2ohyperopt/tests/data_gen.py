from h2o import import_file


def data_gen():
    # 'path' can point to a local file, hdfs, s3, nfs, Hive, directories, etc.
    titanic_df = import_file(path="http://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/titanic.csv")

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
