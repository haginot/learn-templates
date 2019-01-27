from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


class PipelineRandomForestClassifier:
    def __init__(self):
        self.steps = [
            ('pca', PCA()),
            ('rf', RandomForestClassifier())
        ]

        self.pipeline = Pipeline(steps=self.steps)

    def get_classifier(self):
        return self.pipeline
