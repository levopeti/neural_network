import numpy as np
import time
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

np.random.seed(int(time.time()))


class NeuralNet(object):
    def __init__(self, input, output):
        self.learning_rate = 0.1
        self.loss = None
        self.error = None
        self.optimizer = None

        self.X = None
        self.Y = None
        self.Y_am = None        # Y argmax

        self.test_x = None
        self.test_y = None
        self.test_y_am = None   # test_y argmax

        self.batch_size = None
        self.num_batches = None
        self.test_num_batches = None
        self.epochs = None

        self.input = input
        self.output = output

        self.model = None

        self.POOL = True

    def __del__(self):
        pass

    def build_model(self, loss="MSE", optimizer=None, learning_rate=None, batch_size=100):
        print("Build the model...\n")
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.loss = self.loss_func(loss)
        self.error = self.error_func(loss)
        self.optimizer = optimizer

        self.input.set_size_forward(self.batch_size, self.learning_rate, self.optimizer)

    def set_weights(self, individual):
        # self.W = np.reshape(np.array(individual[:7840]), (784, 10))  # shape (784, 10)
        # self.b = np.array(individual[-10:])  # shape (10,)
        pass

    def get_weights_as_genes(self):
        return np.concatenate((np.reshape(self.W, (7840,)), self.b), axis=None)

    def save_weights(self, filepath):
        weights = []
        self.input.save_weights(weights)
        print("Save weights to ", filepath)

        with open(os.path.join(filepath), 'wb') as fp:
            pickle.dump(weights, fp)

    def load_weights(self, filepath):
        with open(os.path.join(filepath), 'rb') as fp:
            weights = pickle.load(fp)

        weights = np.array(weights)
        self.input.load_weights(weights)

    def predict(self, sample):
        """Sample must have shape (1, 1, -1)"""
        self.input.forward_process(sample)
        o = self.output.output
        prediction = self.softmax(o)

        return prediction

    def evaluate(self, test=False):
        """Evaluate the model."""
        global_loss = 0
        predicted_values = []

        if test:
            num_batches = self.test_num_batches
        else:
            num_batches = self.num_batches

        if self.POOL:
            def eval_on_batch(b):
                # print(b)

                # forward process
                start, end = b * self.batch_size, (b + 1) * self.batch_size
                self.forward(start, end, test)

                o = self.output.output

                if not test:
                    if end > self.Y.shape[0]:
                        end = self.Y.shape[0]

                    loss, predicted_value = self.loss(o, self.Y[start:end])
                else:
                    if end > self.test_y.shape[0]:
                        end = self.test_y.shape[0]

                    loss, predicted_value = self.loss(o, self.test_y[start:end])

                # print(loss)
                # print(predicted_value.shape)
                # exit()

                return loss, predicted_value

            with ThreadPoolExecutor(max_workers=cpu_count()) as p:
                results = p.map(eval_on_batch, range(num_batches))

            for loss, predicted_value in results:
                predicted_values.append(predicted_value)
                global_loss += loss
        else:
            for b in range(num_batches):
                # print(b)

                # forward process
                start, end = b * self.batch_size, (b + 1) * self.batch_size
                self.forward(start, end, test)

                o = self.output.output

                if not test:
                    if end > self.Y.shape[0]:
                        end = self.Y.shape[0]

                    loss, predicted_value = self.loss(o, self.Y[start:end])
                else:
                    if end > self.test_y.shape[0]:
                        end = self.test_y.shape[0]

                    loss, predicted_value = self.loss(o, self.test_y[start:end])

                # print(loss)
                # print(predicted_value.shape)
                # exit()
                predicted_values.append(predicted_value)

                global_loss += loss

        predicted_values = np.array(predicted_values).reshape(-1,)

        return global_loss / num_batches, self.accurate_func(np.array(predicted_values), test)

    def train_step(self):
        """Train one epoch on the network with backpropagation."""

        for b in range(self.num_batches):
            # print(b)

            # forward
            start_time = time.time()
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            self.forward(start, end)

            if self.output.z_nesterov is not None:
                o = self.output.act(self.output.z_nesterov)
            else:
                o = self.output.output

            # print("Time of forward: {}s".format(time.time() - start_time))
            error = self.error(o, self.Y[start:end])

            # backward
            start_time = time.time()
            self.backward(error)
            # print("Time of backward: {}s".format(time.time() - start_time))
            # input()

    def forward(self, start, end, test=False):
        if not test:
            if end > self.X.shape[0]:
                end = self.X.shape[0]

            self.input.forward_process(self.X[start: end])
        else:
            if end > self.test_x.shape[0]:
                end = self.test_x.shape[0]

            self.input.forward_process(self.test_x[start: end])

    def backward(self, error):
        # for the first layer the output_bp = error
        self.output.backward_process(error)

    def train(self, train_x, train_y, test_x=None, test_y=None, epochs=100):
        self.X = train_x
        self.Y = train_y
        self.Y_am = np.argmax(train_y.reshape(-1, 10), axis=1)

        self.test_x = test_x
        self.test_y = test_y
        self.test_y_am = np.argmax(test_y.reshape(-1, 10), axis=1)

        self.epochs = epochs
        self.num_batches = self.X.shape[0] // self.batch_size
        self.test_num_batches = self.test_x.shape[0] // self.batch_size

        print("Start training the model...\n")

        for i in range(self.epochs):
            start = time.time()
            self.train_step()
            train_time = time.time() - start

            start = time.time()
            loss_value, accurate = self.evaluate()

            if self.test_x is not None:
                test_loss_value, test_accurate = self.evaluate(test=True)
            else:
                test_loss_value, test_accurate = None, None

            eval_time = time.time() - start

            self.print_stat(i, loss_value, accurate, test_loss_value, test_accurate, train_time, eval_time)

    @staticmethod
    def print_stat(i, loss_value, accurate, test_loss_value, test_accurate, train_time, eval_time):
        if test_loss_value:
            # print("EPOCH", i + 1, " Train acc.: {0:.2f}% ".format(accurate * 100), "Train loss: {0:.4f}  ".format(loss_value), "Test acc.: {0:.2f}% ".format(test_accurate * 100), "Test loss: {0:.4f}  ".format(test_loss_value), "Train/eval time: {0:.2f}s/{1:.2f}s\n".format(train_time, eval_time))
            print("EPOCH", i + 1, " Train/eval acc.: {0:.2f}/{1:.2f}% ".format(accurate * 100, test_accurate * 100), "Train/eval loss: {0:.4f}/{1:.4f}  ".format(loss_value, test_loss_value), "Train/eval time: {0:.2f}s/{1:.2f}s\n".format(train_time, eval_time))

        else:
            print("EPOCH", i + 1, "\tTrain accurate: {0:.2f}%\t".format(accurate * 100), "Train loss: {0:.4f}\t".format(loss_value), "Train/eval time: {0:.2f}s/{1:.2f}s\n".format(train_time, eval_time))

    def accurate_func(self, pred, test=False):
        goal = 0

        if not test:
            for i in range(pred.shape[0]):
                if pred[i] == self.Y_am[i]:        # shape: (-1, 1, num_class) --> shape: (-1, num_class)
                    goal += 1
        else:
            for i in range(pred.shape[0]):
                if pred[i] == self.test_y_am[i]:
                    goal += 1

        return goal / pred.shape[0]

    def loss_func(self, type):
        def mse(o, y):
            return np.square(o - y).sum() * 0.5 / self.batch_size, np.argmax(o, axis=2)

        def xe(o, y):
            prediction = self.softmax(o)
            return self.cross_entropy(prediction, y), np.argmax(prediction, axis=2)

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def error_func(self, type):
        def mse(o, y):
            return np.subtract(o, y)

        def xe(o, y):
            """
            d_cross_entropy
            """
            prediction = self.softmax(o)
            return np.subtract(prediction, y)

        if type == "MSE":
            return mse

        elif type == "XE":
            return xe

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in x.
        https://deepnotes.io/softmax-crossentropy
        Input and Return shape: (batch size, num of class)
        """
        e_x = np.exp(x - np.max(x, axis=2).reshape(self.batch_size, 1, 1))
        ex_sum = np.sum(e_x, axis=2)
        ex_sum = ex_sum.reshape((self.batch_size, 1, 1))

        return e_x / ex_sum

    @staticmethod
    def cross_entropy(p, y):
        """
        p is the prediction after softmax, shape: (batch size, num of class)
        y is labels (one hot vectors), shape: (batch size, num of class)
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        https://deepnotes.io/softmax-crossentropy
        Return size is a scalar.
        cost = -(1.0/m) * np.sum(np.dot(np.log(A), Y.T) + np.dot(np.log(1-A), (1-Y).T))
        """
        # y = y.argmax(axis=1)

        # batch_size
        m = y.shape[0]

        # log_likelihood = -np.log(p[range(m), y])
        # loss = np.sum(log_likelihood) / m

        cost = -(1.0 / m) * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return cost

    @staticmethod
    def d_cross_entropy(p, y):
        """
        p is the prediction after softmax, shape: (batch size, num of class)
        y is labels (one hot vectors), shape: (batch size, num of class)
        Note that y is not one-hot encoded vector.
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        https://deepnotes.io/softmax-crossentropy
        Return shape: (batch size, num of class)
        """
        grad = np.subtract(p, y)
        return grad


if __name__ == "__main__":
    print("Please use the train.py script!")



