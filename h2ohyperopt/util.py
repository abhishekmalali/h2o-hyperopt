from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt


# Introducing a new wrapper for the the hyperopt randint function

def randint(name, (lower, upper)):
    """
    Function to create hyperopt randint variable
    Input
    ------------------
    name - Variable name
    (lower, upper) - Tuple with lower and upper bound
    """
    assert type(lower) == int, "Value for lower bound is not an integer"
    assert type(upper) == int, "Value for upper bound is not an integer"

# Build a selection list
    selList = [i+lower for i in range(upper-lower+1)]
    return hp.choice(name, selList)


def uniform(name, (lower, upper)):
    """
    Function to create hyperopt uniform random variable
    Input
    ------------------
    name - Variable name
    (lower, upper) - Tuple with lower and upper bound
    """
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.uniform(name, lower, upper)


def choice(name, choice_list):
    """
    Function to create hyperopt choice variable
    Input
    ------------------
    name - Variable name
    choice_list - List with possible choices
    """
    assert len(choice_list) > 1, "Error with choices provided"
    return hp.choice(name, choice_list)


def quniform(name, (lower, upper, q)):
    """
    Function to create hyperopt quniform variable
    Input
    ------------------
    name - Variable name
    (lower, upper, q) - Tuple of bounds and q value.

    Distribution looks like round(uniform(lower, upper) / q) * q
    """
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.quniform(name, lower, upper, q)


def loguniform(name, (lower, upper)):
    """
    Function to create hyperopt loguniform variable
    Input
    ------------------
    name - Variable name
    (lower, upper) - Tuple of bounds.

    Distribution looks like exp(uniform(lower, upper))
    """
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.loguniform(name, lower, upper)


def qloguniform(name, (lower, upper, q)):
    """
    Function to create hyperopt qloguniform variable
    Input
    ------------------
    name - Variable name
    (lower, upper, q) - Tuple of bounds and q value.

    Distribution looks like round(exp(uniform(lower, upper)) / q) * q
    """
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.qloguniform(name, lower, upper, q)


def normal(name, (mu, sigma)):
    """
    Function to create hyperopt normal variable
    Input
    ------------------
    name - Variable name
    (mu, sigma) - Tuple of mean and standard deviation.
    """
    return hp.normal(name, mu, sigma)


def qnormal(name, (mu, sigma, q)):
    """
    Function to create hyperopt qnormal variable
    Input
    ------------------
    name - Variable name
    (mu, sigma, q) - Tuple of mean, standard deviation and q value.

    Distribution looks like round(normal(mu, sigma)/ q)* q
    """
    return hp.normal(name, mu, sigma, q)


def lognormal(name, (mu, sigma)):
    """
    Function to create hyperopt lognormal variable
    Input
    ------------------
    name - Variable name
    (mu, sigma, q) - Tuple of mean, standard deviation and q value.

    Distribution looks like exp(normal(mu, sigma))
    """
    return hp.lognormal(name, mu, sigma)


def qlognormal(name, (mu, sigma, q)):
    """
    Function to create hyperopt qlognormal variable
    Input
    ------------------
    name - Variable name
    (mu, sigma, q) - Tuple of mean, standard deviation and q value.

    Distribution looks like round(exp(normal(mu, sigma))/ q)* q
    """
    return hp.qlognormal(name, mu, sigma, q)


def show_distributions_info():
    print "List of Distributions available"
    list_dist = [uniform, randint, choice, loguniform, quniform,
                 qloguniform, normal, lognormal, qlognormal]
    for dist in list_dist:
        print "Distribution name :", dist.__name__
        print "Docstring"
        print "------------------------------------"
        print dist.__doc__
# Additional helped functions
def gen_metric(func, metric):
    if metric == 'auc':
        return func.auc()
    elif metric == 'logloss':
        return func.logloss()
    elif metric == 'mse':
        return func.mse()
    elif metric == 'r2':
        return func.r2()
    else:
        raise ValueError("Error metric not available in H2O")


def update_model_parameters(model, params):
    """Function to update parameters specially for Model Optimizers"""
    base_attrs = dir(model)
    for key in params.keys():
        if key not in base_attrs and key != 'metric':
            raise ValueError("Wrong parameter %s passed to the model"
                             % key)
        else:
            if type(params[key]) is tuple:
                setattr(model, key, list(params[key]))
            else:
                setattr(model, key, params[key])
    return model


def update_model_parameters_GLM(model, params):
    """Function to update parameters specially for GLMOptimizers"""
    base_attrs = dir(model)
    for key in params.keys():
        if key not in base_attrs and key != 'metric':
            raise ValueError("Wrong parameter %s passed to the model"
                             % key)
        else:
            if type(params[key]) is tuple:
                setattr(model, key, list(params[key]))
            elif key in ['alpha', 'lambda_']:
                setattr(model, key, [params[key]])
            else:
                setattr(model, key, params[key])
    return model
