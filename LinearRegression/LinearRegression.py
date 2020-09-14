import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, x: list, y: list):
        if len(x) != len(y):
            raise ValueError("The lists are unequals")

        self.theta0 = 0.
        self.theta1 = 0.
        self.learning_rate = 0.0001  # TODO: Be sure

        self.x = x
        self.y = y
        self.length = len(x)

        self.trained = False

    @staticmethod
    def _hypothesis(theta0, theta1, mileage):
        return theta0 + (theta1 * mileage)

    def _get_sum_theta0(self, theta0, theta1):
        value = 0.
        for i in range(self.length):
            value += self._hypothesis(theta0, theta1, self.x[i]) - self.y[i]
        return value

    def _get_sum_theta1(self, theta0, theta1):
        value = 0.
        for i in range(self.length):
            value += (self._hypothesis(theta0, theta1, self.x[i]) - self.y[i]) * self.x[i]
        return value

    def _gradient_descent(self):
        sum0 = self._get_sum_theta0(self.theta0, self.theta1) / self.length
        sum1 = self._get_sum_theta1(self.theta0, self.theta1) / self.length

        tmp_theta0 = self.learning_rate * sum0
        tmp_theta1 = self.learning_rate * sum1

        self.theta0 = self.theta0 - tmp_theta0
        self.theta1 = self.theta1 - tmp_theta1

        return abs(tmp_theta0) < 0.000001 and abs(tmp_theta1) < 0.000001

    def train_model(self):
        while self._gradient_descent() is False:
            print(self.theta0, self.theta1)
            continue

        self.trained = True


if __name__ == '__main__':
    data = pd.read_csv('../data.csv')
    km = data['km'].tolist()
    price = data['price'].tolist()

    test = LinearRegression(km, price)
