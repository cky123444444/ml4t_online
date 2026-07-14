class Predictor:
    def __init__(self, model):
        self.model = model

    def train_model(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)