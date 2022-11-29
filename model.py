import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import data_processing
class Perceptron:
    w = 0
    b = 0
    def __init__(self):
        self.w = None
        self.b = None

    def model(self, x):
        return 1 if (np.dot(self.w, x) >= self.b) else 0

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y, epochs=1, lr=1):

        self.w = np.ones(X.shape[1])
        self.b = 0

        accuracy = {}
        max_accuracy = 0

        wt_matrix = []

        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                    self.b = self.b - lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)

            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b

        self.w = chkptw
        self.b = chkptb

        return np.array(wt_matrix)

perceptro = Perceptron()
wt_matrix = perceptro.fit(data_processing.X_train, data_processing.Y_train, 10000, 0.5)
Y_pred_test = perceptro.predict(data_processing.X_test)
print(accuracy_score(Y_pred_test, data_processing.Y_test))