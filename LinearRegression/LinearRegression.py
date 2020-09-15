class LinearRegression:

    def __init__(self, x_axis: list, y_axis: list, theta0=0.0, theta1=0.0, learning_rate=0.1):
        if len(x_axis) != len(y_axis):
            raise ValueError("The dataset is invalid. Their length differ")

        self.mean_theta0 = theta0
        self.mean_theta1 = theta1
        self.learning_rate = learning_rate

        self.max_x = max(x_axis)
        self.x = [value / self.max_x for value in x_axis]

        self.max_y = max(y_axis)
        self.y = [value / self.max_y for value in y_axis]

        self.length = len(x_axis)

    @staticmethod
    def hypothesis(theta0, theta1, mileage):
        return theta0 + (theta1 * mileage)

    @property
    def theta0(self):
        return self.mean_theta0 * self.max_y

    @property
    def theta1(self):
        return self.mean_theta1 * (self.max_y / self.max_x)

    # def _get_sum_theta0(self, theta0, theta1):
    #     value = 0.
    #     for i in range(self.length):
    #         value += self.hypothesis(theta0, theta1, self.x[i]) - self.y[i]
    #     return value
    #
    # def _get_sum_theta1(self, theta0, theta1):
    #     value = 0.
    #     for i in range(self.length):
    #         value += (self.hypothesis(theta0, theta1, self.x[i]) - self.y[i]) * self.x[i]
    #     return value

    def _gradient_descent(self):
        # sum0 = self._get_sum_theta0(self.mean_theta0, self.mean_theta1) / self.length
        # sum1 = self._get_sum_theta1(self.mean_theta0, self.mean_theta1) / self.length

        tmp_theta0 = self.learning_rate * (1 / self.length) * \
            sum(self.hypothesis(self.mean_theta0, self.mean_theta1, self.x[i]) - self.y[i]
                for i in range(self.length))

        tmp_theta1 = self.learning_rate * (1 / self.length) * \
            sum((self.hypothesis(self.mean_theta0, self.mean_theta1, self.x[i]) - self.y[i]) * self.x[i]
                for i in range(self.length))

        self.mean_theta0 = self.mean_theta0 - tmp_theta0
        self.mean_theta1 = self.mean_theta1 - tmp_theta1

        return abs(tmp_theta0) < 0.000001 and abs(tmp_theta1) < 0.000001

    def train_model(self):
        while self._gradient_descent() is False:
            continue

    # def guess_price(self, mileage: int) -> str:
    #     if self.trained is False:
    #         print("Model is not trained. Starting training..")
    #         self.train_model()
    #         print("Done.", "", sep='\n')
    #
    #     theta0 = self.theta0 * self.max_y
    #     theta1 = self.theta1 * (self.max_y / self.max_x)
    #     price = int(self.hypothesis(theta0, theta1, mileage))
    #
    #     return f"According to this set of data, the price of a car " \
    #         f"that has been drove for {mileage} miles is around ${price}"
