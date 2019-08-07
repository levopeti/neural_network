import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, (-1, 28 * 28))
x_test = np.reshape(x_test, (-1, 28 * 28))

X = np.array(np.append(x_train, x_test, axis=0))
Y = np.eye(10)[np.append(y_train, y_test)]  # one hot vectors

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784, ), activation=tf.nn.softmax)
    # tf.keras.layers.Dense(256, input_shape=(784, ), activation=tf.nn.sigmoid),
    # tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, epochs=10)
loss, acc = model.evaluate(X, Y)


