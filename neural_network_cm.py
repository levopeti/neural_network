import numpy as np
import tensorflow as tf
import time
import pickle
import os
import cudamat as cm


cm.cuda_set_device(0)
np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self):
        self.num_class = 10
        self.learning_rate = -0.1

        loss = "MSE"
        self.loss = self.loss_func(loss)
        self.error = self.error_func(loss)

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = np.reshape(x_train, (-1, 1, 28 * 28))
        x_test = np.reshape(x_test, (-1, 1, 28 * 28))

        self.X = np.array(np.append(x_train, x_test, axis=0))
        self.Y = np.eye(self.num_class)[np.append(y_train, y_test)]  # one hot vectors

        self.input_length = len(self.X[0, 0])
        self.train_data_length = len(self.X)

        cm.init()
        self.output_bp = None

        self.model = []
        self.build_model()

    def __del__(self):
        cm.shutdown()

    def build_model(self):
        print("Build the model...\n")
        self.model.append(Layer("fc", self.input_length, 10, self.learning_rate, "sigmoid"))
        # self.model.append(Layer("fc", 256, 10, self.learning_rate, "sigmoid"))

    def set_weights(self, individual):
        self.W = np.reshape(np.array(individual[:7840]), (784, 10))  # shape (784, 10)
        self.b = np.array(individual[-10:])  # shape (10,)

    def get_weights_as_genes(self):
        return np.concatenate((np.reshape(self.W, (7840,)), self.b), axis=None)

    def save_weights(self):
        weights = []
        for layer in self.model:
            weights.append(layer.W)
            weights.append(layer.b)

        with open(os.path.join('weights.txt'), 'wb') as fp:
            pickle.dump(weights, fp)

    def evaluate(self):
        """Evaluate the model."""
        global_loss = 0
        predicted_values = []

        for i in range(self.train_data_length):

            # forward
            o = self.forward(i)
            o.copy_to_host()

            loss, predicted_value = self.loss(o.numpy_array, self.Y[i])
            predicted_values.append(predicted_value)

            global_loss += loss

        return global_loss, self.accurate_func(np.array(predicted_values))

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        for i in range(self.train_data_length):
            if i % 10000 == 0:
                print(i)

            # forward
            start = time.time()
            o = self.forward(i)

            #print("Time of forward: {}s\n".format(time.time() - start))
            error = self.error(o.asarray(), self.Y[i])

            # backward
            start = time.time()
            self.backward(error, i)
            #print("Time of backward: {}s\n".format(time.time() - start))
            #input()

    def forward(self, i):
        start = time.time()
        data = cm.CUDAMatrix(self.X[i])
        # print("Time of forward0: {}s".format(time.time() - start))
        for layer in self.model:
            data = layer.forward(data)
        return data

    def backward(self, output_bp, i):
        # for the first layer the output_bp = error
        start = time.time()
        self.output_bp = cm.CUDAMatrix(output_bp)
        data = cm.CUDAMatrix(self.X[i])
        # print("Time of backward0: {}s".format(time.time() - start))
        for j in range(len(self.model))[::-1]:
            layer = self.model[j]
            if j == 0:
                self.output_bp = layer.backward(data, self.output_bp)
            else:
                self.output_bp = layer.backward(self.model[j - 1].output, self.output_bp)

    def base_line(self, epochs):
        print("Start training the model...\n")

        for i in range(epochs):
            start = time.time()
            self.train_step()
            loss_value, accurate = self.evaluate()
            # print("max of W: {0:.2f}".format(np.amax(self.W)))
            # print("min of W: {0:.2f}\n".format(np.amin(self.W)))
            # print("max of b: {0:.2f}".format(np.amax(self.b)))
            # print("min of b: {0:.2f}\n".format(np.amin(self.b)))

            print("EPOCH", i + 1, "\tAccurate: {0:.2f}%\t".format(accurate * 100), "Loss: {0:.2f}\t".format(loss_value), "ETA: {0:.2f}s\n".format(time.time() - start))
            if i == 30:
                self.learning_rate *= 0.5

    def accurate_func(self, pred):
        goal = 0

        for i in range(pred.shape[0]):

            if pred[i] == np.argmax(self.Y[i]):
                goal += 1
        return goal / pred.shape[0]

    def loss_func(self, type):
        def mse(o, y):
            return np.square(o - y).sum() * 0.5, np.argmax(o)

        def xe(o, y):
            return self.cross_entropy(o, y), np.argmax(self.softmax(o))

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def error_func(self, type):
        def mse(o, y):
            return np.subtract(o, y)

        def xe(o, y):
            return self.d_cross_entropy(o, y)[0]

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def cross_entropy(self, x, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = np.array(y).reshape((1, -1))
        x = np.array(x).reshape((1, -1))
        y = y.argmax(axis=1)
        m = y.shape[0]
        p = self.softmax(x)
        # We use multidimensional array indexing to extract
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[range(m), y])
        loss = np.sum(log_likelihood) / m
        return loss

    def d_cross_entropy(self, x, y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        y = np.array(y).reshape((1, -1))
        x = np.array(x).reshape((1, -1))
        y = y.argmax(axis=1)
        m = y.shape[0]
        grad = self.softmax(x)
        grad[range(m), y] -= 1
        grad = grad / m
        return grad


class Layer(object):
    def __init__(self, layer_type, input_size, output_size, learning_rate, activation):
        self.layer_type = layer_type
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.act = self.act_func(activation)
        self.d_act = self.d_act_func(activation)

        self.output = None

        self.W = None
        self.b = None
        self.z = None

        self.forward = None
        self.backward = None

        if self.layer_type == "fc":
            self.W = np.random.rand(self.input_size, self.output_size) * 1 - 0.5
            self.b = np.random.rand(1, self.output_size) * 1 - 0.5

            self.W = cm.CUDAMatrix(self.W)
            self.b = cm.CUDAMatrix(self.b)
            self.z = cm.empty((1, self.output_size))
            self.d_act_z = cm.empty((1, self.output_size))
            self.output = cm.empty((1, self.output_size))
            self.delta_b = cm.empty((1, self.output_size))
            self.delta_W = cm.empty((self.input_size, self.output_size))
            self.output_bp = None  # cm.empty((1, self.input_size))

            self.forward = self.dense_fw
            self.backward = self.dense_bw

    def dense_fw(self, x):
        """Fully connected layer forward process."""
        start = time.time()
        self.b.add(cm.dot(x, self.W), target=self.z)
        # print("Time of forward1: {}s".format(time.time() - start))

        start = time.time()
        output = self.act(self.z.asarray())
        self.output = cm.CUDAMatrix(output)
        # print("Time of forward2: {}s\n".format(time.time() - start))

        return self.output

    def dense_bw(self, x, input_error):
        """Fully connected layer backward process"""
        start = time.time()
        self.d_act_z.copy_to_host()
        self.d_act_z.numpy_array = self.d_act(self.z.asarray())
        self.d_act_z.copy_to_device()
        # print("Time of backward1: {}s".format(time.time() - start))

        start = time.time()
        # input_error = cm.CUDAMatrix(input_error)
        input_error.mult(self.d_act_z, target=self.delta_b)

        x = x.transpose()
        self.delta_W = cm.dot(x, self.delta_b)

        self.output_bp = cm.dot(self.delta_b, self.W.transpose())
        # print("Time of backward2: {}s".format(time.time() - start))

        start = time.time()
        self.W.add(self.delta_W.mult(self.learning_rate), target=self.W)
        self.b.add(self.delta_b.mult(self.learning_rate), target=self.b)
        # print("Time of backward3: {}s\n".format(time.time() - start))

        return self.output_bp

    def act_func(self, type):
        if type == "tanh":
            return self.tanh

        elif type == "sigmoid":
            return self.log

        elif type == "relu":
            return self.relu

    def d_act_func(self, type):
        if type == "tanh":
            return self.d_tanh

        elif type == "sigmoid":
            return self.d_log

        elif type == "relu":
            return self.d_relu

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def d_tanh(x):
        return 1 - np.tanh(x) ** 2

    def log(self, x):
        return 1 / (1 + np.exp(-1 * x))
        # self.tmp1.assign(x)
        # return cm.pow(self.tmp1.add(cm.exp(self.tmp1.mult(-1)), target=self.tmp1), -1)

    def d_log(self, x):
        return self.log(x) * (1 - self.log(x))
        # self.tmp1.assign(self.log(x))
        # self.tmp2.assign(self.log(x))
        # self.tmp1.mult(-1)
        # self.tmp1.add(1, target=self.tmp1)
        # self.tmp2.mult(self.tmp1, target=self.tmp2)
        # return self.tmp2

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def d_relu(x):
        return np.where(x > 0, 1, 0)


if __name__ == "__main__":
    nn = NeuralNet()
    nn.base_line(60)
    nn.save_weights()