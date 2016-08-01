from unittest import TestCase
import h2ohyperopt

class TestModelOptMetrics(TestCase):
    def testCheckMetricSet(self):
        model = h2ohyperopt.ModelOptimizer(metric='auc')
        assert model.check_if_metric_set() is True

    def testSetMetric(self):
        model = h2ohyperopt.ModelOptimizer()
        assert model.check_if_metric_set() is False
        model.set_metric('auc')
        assert model.check_if_metric_set() is True
