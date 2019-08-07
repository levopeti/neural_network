# import numpy as np
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../../')

from research.neural_network.layer import Input, Dense, Conv2d, Pool2d, Flatten, Concat, Add
from research.neural_network.neural_network import NeuralNet

nn = None
gray_image = None


def build_nn():
    global nn

    num_class = 10

    ip = Input(input_size=(1, 784))
    # x = Conv2d(number_of_kernel=3, kernel_size=5, activation="relu")(ip)
    # x = Pool2d(kernel_size=5)(x)
    # y = Conv2d(number_of_kernel=3, kernel_size=5, activation="relu")(ip)
    # y = Pool2d(kernel_size=5)(y)
    # a = Add(weights_of_layers=[1, 3])([x, y])
    # c = Concat(axis=1)([x, y])
    # f = Flatten()(a)
    x1 = Dense(units=256, activation="sigmoid")(ip)

    # y1 = Dense(units=20, activation="sigmoid")(x1)
    # y2 = Dense(units=20, activation="sigmoid", learning_rate=1)(x1)
    # c1 = Concat(axis=1)([y1, y2])
    #
    # x2 = Dense(units=50, activation="sigmoid", learning_rate=1)(ip)
    # z1 = Dense(units=20, activation="sigmoid", learning_rate=1)(x2)
    # z2 = Dense(units=20, activation="sigmoid", learning_rate=1)(x2)
    # c2 = Concat(axis=1)([z1, z2])

    # c = Concat(axis=1)([c1, c2])
    op = Dense(units=num_class, activation="sigmoid")(x1)

    nn = NeuralNet(ip, op)
    nn.build_model(loss="XE", learning_rate=0.1, batch_size=1)
    nn.load_weights(filepath='weights.txt')


def load_image(path):
    global gray_image

    ii = cv2.imread(path)
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image[30:-30, 130:-30]
    gray_image = cv2.resize(gray_image, (28, 28))
    # plt.imshow(gray_image, cmap='Greys')
    # plt.show()
    gray_image = gray_image.reshape(1, 1, -1)
    # print(gray_image.shape)


def predict(path):
    global nn, gray_image

    if nn:
        load_image(path)
        prediction = nn.predict(gray_image)
    else:
        load_image(path)
        build_nn()
        prediction = nn.predict(gray_image)

    return prediction


if __name__ == "__main__":
    path = '/home/biot/projects/kivy/digit_rec/test0001.jpg'
    predict(path)



