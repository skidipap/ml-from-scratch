import numpy as np 

class LinearRegression:
    def __init__(self, alpha=0.001, n_iters=1000, loss_function="MSE"):
        self.alpha = alpha
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_function = loss_function

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = 0.01 * np.random.randn(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients

            if self.loss_function == "MSE":
                loss = (1 / n_samples) * np.sum((y - y_predicted) ** 2)
                dw = - (2 / n_samples) * np.sum((y - y_predicted) * X)
                db = - (2 / n_samples) * np.sum(y - y_predicted)
            
            if self.loss_function == "MAE": 
                loss = (1 / n_samples) * np.sum(abs(y - y_predicted))
                
                dw = - X * (np.sum(y - y_predicted)) / abs(np.sum(y - y_predicted))
                db = - (np.sum(y - y_predicted)) / abs(np.sum(y - y_predicted))
            
            # update parameters
            self.weights +=  - self.alpha * dw
            self.bias += - self.alpha * db
            

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated

