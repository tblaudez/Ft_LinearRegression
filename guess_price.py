#!/usr/bin/env python

from __future__ import with_statement

import matplotlib.pyplot as plt
import numpy as np
import pickle
from argparse import ArgumentParser

from GetData import get_data
from LinearRegression import LinearRegression


def parse_args():
    parser = ArgumentParser(description="This script deduce the price of a car "
                                        "according to it's mileage using a set of organized data")

    parser.add_argument('-g', '--graph', help="display a neat graph containing all the data to help you understand",
                        action='store_true', dest='graph')

    return parser.parse_args()


def display_graph(a, b, cur_km, cur_price):
    km_data, price_data = get_data()

    x_axis = np.linspace(min(km_data), max(max(km_data), cur_km))
    y_axis = a * x_axis + b

    plt.plot(km_data, price_data, 'x', label='Price for Mileage', color='blue')
    plt.plot(x_axis, y_axis, '-r', label='Linear regression', color='green')
    plt.plot(cur_km, cur_price, '^', label="You are here", color='red')

    plt.title('Price of a car according to its mileage')
    plt.xlabel('Mileage', color='#1C2833')
    plt.ylabel('Price', color='#1C2833')
    plt.legend(loc='upper right')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    args = parse_args()
    graph = args.graph

    try:
        with open('theta0.save', 'rb') as f:
            theta0 = pickle.load(f)
    except FileNotFoundError:
        print("No saved value for theta0, assuming 0")
        theta0 = 0.0

    try:
        with open('theta1.save', 'rb') as f:
            theta1 = pickle.load(f)
    except FileNotFoundError:
        print("No saved value for theta1, assuming 0")
        theta1 = 0.0

    mileage = abs(int(input("For how many kilometers has your car been drove ? ")))
    price = int(LinearRegression.hypothesis(theta0, theta1, mileage))
    print(f"According to the data. A car that drove for {mileage} miles can be sold for about ${price}")

    if graph is True:
        display_graph(theta1, theta0, mileage, price)

