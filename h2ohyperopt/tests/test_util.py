import h2o
from unittest import TestCase
import h2ohyperopt
from data_gen import *
h2o.init()


class TestUtil(TestCase):
    def testUtilBasic(self):
        quni = h2ohyperopt.util.quniform('test', (0, 1, 0.5))
        loguni = h2ohyperopt.util.loguniform('test', (0, 1))
        qloguni = h2ohyperopt.util.qloguniform('test', (0, 1, 0.5))
        norm = h2ohyperopt.normal('test', (0, 0.1))
        qnorm = h2ohyperopt.util.qnormal('test', (0, 0.1, 0.5))
        lognorm = h2ohyperopt.lognormal('test', (0, 0.1))
        qlognorm = h2ohyperopt.util.qlognormal('test', (0, 0.1, 0.5))
