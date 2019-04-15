import pickle

class Metric_Store:
    def __init__(self, load=None):
        self.clear()

    def clear(self):
        self.training_loss = []
        self.test_loss = []

    def log(self, training_loss, test_loss):
        self.training_loss.append(training_loss)
        self.test_loss.append(test_loss)

    def save(self, path):
        with open(path, 'wb') as pfile:
            pickle.dump(self, pfile)

    def load(self, path):
        with open(path, 'rb') as pfile:
            data = pickle.load(pfile)
            self.training_loss = data.training_loss
            self.test_loss = data.test_loss
