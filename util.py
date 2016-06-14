from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt


#Introducing a new wrapper for the the hyperopt randint function

def randint(name, lower, upper):
    assert type(lower) == int, "Value for lower bound is not an integer"
    assert type(upper) == int, "Value for upper bound is not an integer"

    #Build a selection list
    selList = [i+lower for i in range(upper-lower+1)]
    return hp.choice(name, selList)

"""
if __name__ == "__main__":
    print randint('a', 10, 20)
"""

