from LinearRegression import LinearRegression
from GetData import get_data
import pickle

if __name__ == '__main__':
    print("Reading data..")
    km, price = get_data()

    print("Creating model..")
    model = LinearRegression(km, price)

    print("Training model..")
    model.train_model()

    print("Saving thetas...")
    with open('theta0.save', 'wb+') as f:
        pickle.dump(model.theta0, f)
    with open('theta1.save', 'wb+') as f:
        pickle.dump(model.theta1, f)

    print("Done.")
