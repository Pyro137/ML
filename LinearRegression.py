import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


#This method will work with only numerical target and features.
class LinearRegression:
    def __init__(self, dataset: pd.DataFrame, target_name: str, learning_rate: float = 0.001, iterations: int = 50):
        self.dataset = dataset
        self.target_name = target_name
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        if dataset[self.target_name].dtype!="int" or y.dtype!="float":
            raise ValueError("Gardaşım bu ne laaaaa heeğğ")
            
    def train_test_split(self):
        df = self.dataset.select_dtypes(include=["float", "int"])
        X = df.drop(self.target_name, axis=1)
        y = df[self.target_name]
            
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train.values, X_test.values, Y_train.values, Y_test.values

    def mean_squared_error(self, y_actual, y_pred):
        return np.mean((y_actual - y_pred) ** 2)

    def linear_regression_model(self):
        X_train, X_test, Y_train, Y_test = self.train_test_split()
        n_example, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.iterations):
            #prediction and train cost
            y_pred = np.dot(X_train, self.weights) + self.bias
            cost_train = self.mean_squared_error(Y_train, y_pred)

            # gradient calculation
            dw = (1 / n_example) * np.dot(X_train.T, (y_pred - Y_train))
            db = (1 / n_example) * np.sum(y_pred - Y_train)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # prediction on test data and test cost
            y_pred_test = np.dot(X_test, self.weights) + self.bias
            cost_test = self.mean_squared_error(Y_test, y_pred_test)

            # save history
            self.cost_history.append(cost_train)
            print(f"Iteration: {i}, Cost (Train): {cost_train:.4f}, Cost (Test): {cost_test:.4f}")

        return y_pred_test

if __name__ == '__main__':
    data = pd.read_csv("datasets/Housing.csv")
    lr = LinearRegression(dataset=data, target_name="mainroad", learning_rate=0.01, iterations=15)
    predictions = lr.linear_regression_model()
    print("Final Predictions:", predictions)
