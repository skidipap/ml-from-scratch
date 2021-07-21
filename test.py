import numpy as np 

from sklearn.model_selection import train_test_split 
from sklearn import datasets 


X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

from mylinear import LinearRegression 


regressor = LinearRegression(alpha=0.001, n_iters = 1000, loss_function="MAE")
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

def mse(y_true, y_predicted):
    result =  np.mean((y_true- y_predicted)**2)
    
    return result 


mse_value= mse(y_test, predictions)

print(mse_value)
