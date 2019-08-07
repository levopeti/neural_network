import tensorflow as tf
from layer import Input, Dense, Conv2d, Pool2d, Flatten, Concat, Add
import numpy as np
from neural_network_cpu import NeuralNet

import optimizers


def train():
    num_class = 10
    mnist = tf.keras.datasets.mnist
    fashion = False

    if fashion:
        from keras.datasets import fashion_mnist

        mnist = fashion_mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (-1, 1, 28 * 28))
    x_test = np.reshape(x_test, (-1, 1, 28 * 28))
    # x_train = np.reshape(x_train, (-1, 1, 28, 28))
    # x_test = np.reshape(x_test, (-1, 1, 28, 28))

    X = np.array(np.append(x_train, x_test, axis=0))
    Y = np.eye(num_class)[np.append(y_train, y_test)].reshape(-1, 1, 10)  # one hot vectors shape: (70000, 1, 10)

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
    sgd = optimizers.SGD(gamma=0.9, nesterov=False)
    adagrad = optimizers.Adagrad()
    adadelta = optimizers.Adadelta()
    rmsprop = optimizers.RMSProp(first_order_momentum=False, gamma=0)

    nn.build_model(loss="XE", optimizer=rmsprop, learning_rate=None, batch_size=100)
    nn.train(train_x=X[:60000], train_y=Y[:60000], test_x=X[60000:], test_y=Y[60000:], epochs=10)
    # nn.save_weights(filepath='weights.txt')


if __name__ == "__main__":
    train()



