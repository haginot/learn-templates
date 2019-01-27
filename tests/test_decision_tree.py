import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from learn_templates.pipeline_random_forest_classifier import PipelineRandomForestClassifier


class TestDecisionTree(unittest.TestCase):
    def test_decision_tree(self):
        clf = PipelineRandomForestClassifier().get_classifier()
        iris = load_iris()
        res = cross_val_score(clf, iris.data, iris.target, cv=10)
        print(res)


if __name__ == '__main__':
    unittest.main()
