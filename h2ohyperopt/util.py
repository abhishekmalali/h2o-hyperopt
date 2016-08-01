from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt


# Introducing a new wrapper for the the hyperopt randint function

def randint(name, (lower, upper)):
    assert type(lower) == int, "Value for lower bound is not an integer"
    assert type(upper) == int, "Value for upper bound is not an integer"

# Build a selection list
    selList = [i+lower for i in range(upper-lower+1)]
    return hp.choice(name, selList)


def uniform(name, (lower, upper)):
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.uniform(name, lower, upper)


def choice(name, choice_list):
    assert len(choice_list) > 1, "Error with choices provided"
    return hp.choice(name, choice_list)


def quniform(name, (lower, upper, q)):
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.quniform(name, lower, upper, q)


def loguniform(name, (lower, upper)):
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.uniform(name, lower, upper)


def qloguniform(name, (lower, upper, q)):
    assert lower < upper, "Lower bound of uniform greater than upper bound"
    return hp.qloguniform(name, lower, upper, q)


def normal(name, (mu, sigma)):
    return hp.normal(name, mu, sigma)


def qnormal(name, (mu, sigma, q)):
    return hp.normal(name, mu, sigma, q)


def lognormal(name, (mu, sigma)):
    return hp.lognormal(name, mu, sigma)


def qlognormal(name, (mu, sigma, q)):
    return hp.qlognormal(name, mu, sigma, q)


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
    base_attrs = dir(model)
    for key in params.keys():
        if key not in base_attrs and key != 'metric':
            raise ValueError("Wrong parameter %s passed to the model"
                             % key)
        else:
            setattr(model, key, params[key])
    return model

"""
if __name__ == "__main__":
    print randint('a', 10, 20)
"""
