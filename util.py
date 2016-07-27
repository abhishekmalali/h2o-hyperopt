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
"""
if __name__ == "__main__":
    print randint('a', 10, 20)
"""
