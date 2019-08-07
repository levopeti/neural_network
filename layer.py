import numpy as np
import time

from abc import ABC, abstractmethod

np.random.seed(int(time.time()))


class Input(object):
    def __init__(self, input_size: tuple):
        self.input_size = input_size
        self.output_size = None
        self.batch_size = None

        self.output = None
        self.next_layer = []

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        output_size = [batch_size]

        for size in self.input_size:
            output_size.append(size)

        self.output_size = tuple(output_size)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def set_next_layer(self, layer):
        self.next_layer.append(layer)

    def forward_process(self, x):
        self.output = x
        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        pass

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)


class Layer(ABC):
    def __init__(self, activation="sigmoid", learning_rate=None, prev_layer=None):
        self.input_size = None
        self.output_size = None
        self.batch_size = None
        self.learning_rate = learning_rate

        self.act = self.act_func(activation)
        self.d_act = self.d_act_func(activation)

        self.output = None
        self.output_bp = None

        self.W = None
        self.b = None
        self.z = None
        self.z_nesterov = None

        # cache for optimizer
        self.cache_W = None
        self.cache_b = None

        self.prev_layer = prev_layer
        self.next_layer = []

        self.prev_layer_set_next_layer()

        self.optimizer = None

    @abstractmethod
    def set_size_forward(self, batch_size, learning_rate, optimizer):
        pass

    @abstractmethod
    def forward_process(self):
        pass

    @abstractmethod
    def backward_process(self, input_error):
        pass

    @abstractmethod
    def save_weights(self, w_array):
        pass

    @abstractmethod
    def load_weights(self, w_array):
        pass

    def set_next_layer(self, layer):
        self.next_layer.append(layer)

    def prev_layer_set_next_layer(self):
        self.prev_layer.set_next_layer(self)

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

    @staticmethod
    def log(x):
        """Numerically stable sigmoid function."""
        return np.where(x > -1e2, 1 / (1 + np.exp(-x)), 1 / (1 + np.exp(1e2)))

    def d_log(self, x):
        return self.log(x) * (1 - self.log(x))

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def d_relu(x):
        return np.where(x > 0, 1, 0)

    def update_weights(self, delta_W, delta_b):
        """Update weights and velocity of the weights."""
        self.W, self.b, self.cache_W, self.cache_b = self.optimizer.run(self.W, self.cache_W, delta_W, self.b,
                                                                        self.cache_b, delta_b, self.learning_rate)


class Dense(Layer):
    def __new__(cls, units, activation="sigmoid", learning_rate=None):
        def set_prev_layer(layer):
            instance = super(Dense, cls).__new__(cls)
            instance.__init__(units, activation=activation, learning_rate=learning_rate, prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, units, activation="sigmoid", learning_rate=None, prev_layer=None):
        super().__init__(activation=activation, learning_rate=learning_rate, prev_layer=prev_layer)
        self.units = units

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.input_size = self.prev_layer.output_size
        self.output_size = (self.batch_size, 1, self.units)

        if self.learning_rate is None:
            self.learning_rate = learning_rate

        self.W = (np.random.rand(self.input_size[2], self.output_size[2]) * 1) - 0.5
        self.b = np.zeros(self.output_size[2])

        self.cache_W, self.cache_b = self.optimizer.init(self.W.shape, self.b.shape)

        log = "Dense layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(self.W.size + self.b.size,
                                                                                          self.input_size,
                                                                                          self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def save_weights(self, w_array):
        w_array.append(self.W)
        w_array.append(self.b)

        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        assert w_array[0].shape == self.W.shape and w_array[1].shape == self.b.shape

        self.W = w_array[0]
        self.b = w_array[1]

        w_array = w_array[2:]

        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Fully connected layer forward process."""
        x = self.prev_layer.output
        self.z = np.add(np.dot(x, self.W), self.b)

        if self.optimizer.name == "SGD":
            if self.optimizer.nesterov:
                nesterov_W = np.subtract(self.W, self.optimizer.gamma * self.cache_W)
                nesterov_b = np.subtract(self.b, self.optimizer.gamma * self.cache_b)
                self.z_nesterov = np.add(np.dot(x, nesterov_W), nesterov_b)

        self.output = self.act(self.z)

        assert self.output.shape == self.output_size

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Fully connected layer backward process"""
        if self.z_nesterov is not None:
            d_act_z = self.d_act(self.z_nesterov)
        else:
            d_act_z = self.d_act(self.z)

        delta_b = np.multiply(input_error, d_act_z)
        x = self.prev_layer.output
        self.output_bp = np.dot(delta_b, np.transpose(self.W))

        delta_W = np.tensordot(x, delta_b, axes=([0, 1], [0, 1]))
        delta_b = np.sum(delta_b, axis=0).reshape(-1)

        # normalization
        delta_W = delta_W / self.batch_size
        delta_b = delta_b / self.batch_size

        self.update_weights(delta_W, delta_b)

        assert self.output_bp.shape == self.input_size
        self.prev_layer.backward_process(self.output_bp)


class Conv2d(Layer):
    def __new__(cls, number_of_kernel, kernel_size, activation="sigmoid", learning_rate=None, mode="valid"):
        def set_prev_layer(layer):
            instance = super(Conv2d, cls).__new__(cls)
            instance.__init__(number_of_kernel=number_of_kernel, kernel_size=kernel_size, activation=activation,
                              learning_rate=learning_rate, prev_layer=layer, mode=mode)
            return instance
        return set_prev_layer

    def __init__(self, number_of_kernel, kernel_size, activation="sigmoid", learning_rate=None, prev_layer=None, mode="valid"):
        super().__init__(activation=activation, learning_rate=learning_rate, prev_layer=prev_layer)
        self.kernel_size = kernel_size
        self.number_of_kernel = number_of_kernel
        self.mode = mode

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.input_size = self.prev_layer.output_size

        if self.learning_rate is None:
            self.learning_rate = learning_rate

        if self.mode == "valid":
            # with 'valid' convolution
            self.output_size = (self.batch_size, self.number_of_kernel,
                                self.input_size[2] - (self.kernel_size[0] - 1),
                                self.input_size[3] - (self.kernel_size[1] - 1))
        if self.mode == "same":
            # with 'same' convolution
            self.output_size = (self.batch_size, self.number_of_kernel,
                                self.input_size[2],
                                self.input_size[3])

        depth = self.input_size[1]
        self.W = (np.random.rand(self.number_of_kernel, depth, self.kernel_size[0], self.kernel_size[1]) * 1) - 0.5
        self.b = np.zeros(self.number_of_kernel)

        self.cache_W, self.cache_b = self.optimizer.init(self.W.shape, self.b.shape)

        log = "2D convolution layer with {} parameters.\nInput size: {}\nOutput size: {}\n"\
            .format(self.W.size + self.b.size, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def save_weights(self, w_array):
        w_array.append(self.W)
        w_array.append(self.b)

        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        assert w_array[0].shape == self.W.shape and w_array[1].shape == self.b.shape

        self.W = w_array[0]
        self.b = w_array[1]

        w_array = w_array[2:]

        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """2d convolution layer forward process."""

        # calculate convolution between self.W and input then the result put in self.z
        self.convolution2d(forward=True, mode=self.mode)

        # add biases
        for kernel_index in range(self.W.shape[0]):
            np.add(self.z[:, kernel_index], self.b[kernel_index], out=self.z[:, kernel_index])

        self.output = self.act(self.z)

        assert self.output.shape == self.output_size, (self.output.shape, self.output_size)
        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """
        2d convolution layer backward process is based on:
        https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf
        https://becominghuman.ai/only-numpy-implementing-convolutional-neural-network-using-numpy-deriving-forward-feed-and-back-458a5250d6e4
        """
        # input_error.shape, d_act_z.shape, delta_batch.shape: self.output_size
        d_act_z = self.d_act(self.z)
        delta_batch = np.multiply(input_error, d_act_z)
        self.output_bp = np.zeros(self.input_size)

        # calculate convolution between self.W and delta_batch then the result put in self.output_bp
        mode = None
        if self.mode == "valid":
            mode = "full"
        if self.mode == "same":
            mode = "same"
        self.convolution2d(image_batch=delta_batch, forward=False, mode=mode)

        # calculate convolution between input and delta_batch
        delta_W = self.convolution2d(image_batch=delta_batch, forward=False, d_W=True, mode="valid")
        delta_b = np.sum(delta_batch, axis=(3, 2, 0))

        # normalization
        delta_W = np.sum(delta_W, axis=0)
        delta_W = delta_W / self.batch_size
        delta_b = delta_b / self.batch_size

        self.update_weights(delta_W, delta_b)
        assert self.output_bp.shape == self.input_size, (self.output_bp.shape, self.input_size)

        self.prev_layer.backward_process(self.output_bp)

    def convolution2d(self, image_batch=None, forward=True, d_W=False, mode="valid"):
        """
        This function is based on:
        https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
        https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf
        """
        if forward:
            image_batch = self.prev_layer.output
            kernelss = self.W
        else:
            if d_W:
                kernelss = np.rot90(self.prev_layer.output, k=2, axes=(2, 3))
            else:
                kernelss = np.rot90(self.W, k=2, axes=(2, 3))

        # number of columns and rows of the input
        I_row_num, I_col_num = image_batch.shape[-2], image_batch.shape[-1]

        # number of columns and rows of the filter
        F_row_num, F_col_num = kernelss.shape[-2], kernelss.shape[-1]

        #  calculate the output dimensions
        output_row_num = I_row_num + F_row_num - 1
        output_col_num = I_col_num + F_col_num - 1
        out_shape = (output_row_num, output_col_num)
        # print(out_shape)

        # zero pad the filter
        padded_kernelss = np.pad(kernelss, ((0, 0), (0, 0), (output_row_num - F_row_num, 0),
                                            (0, output_col_num - F_col_num)),
                                 'constant', constant_values=0)

        def toeplitz_test(c, r):
            c = np.asarray(c).ravel()
            r = np.asarray(r).ravel()
            # Form a 1D array of values to be used in the matrix, containing a reversed
            # copy of r[1:], followed by c.
            vals = np.concatenate((r[-1:0:-1], c))
            a, b = np.ogrid[0:len(c), len(r) - 1:-1:-1]
            indx = a + b
            # `indx` is a 2D array of indices into the 1D array `vals`, arranged so
            # that `vals[indx]` is the Toeplitz matrix.
            return vals[indx], indx

        c_2 = range(1, padded_kernelss.shape[2] + 1)
        r_2 = np.r_[c_2[0], np.zeros(I_row_num - 1, dtype=int)]
        doubly_indices, _ = toeplitz_test(c_2, r_2)

        # creat doubly blocked matrix with zero values
        toeplitz_m, indexes = toeplitz_test(padded_kernelss[0, 0, 0, :], np.r_[padded_kernelss[0, 0, 0, :][0], np.zeros(I_col_num - 1)])
        toeplitz_shape = toeplitz_m.shape
        h = toeplitz_shape[0] * doubly_indices.shape[0]
        w = toeplitz_shape[1] * doubly_indices.shape[1]
        doubly_blocked_shape = [h, w]
        np_matmul = np.matmul

        # tile toeplitz matrices for each row in the doubly blocked matrix
        b_h, b_w = toeplitz_shape  # hight and withs of each block

        # vectorization of image_batch
        image_batch = image_batch[:, :, ::-1, :]
        vectorized_image_batch = np.reshape(image_batch, (image_batch.shape[0], image_batch.shape[1],
                                                          image_batch.shape[2] * image_batch.shape[3]))

        result = np.zeros((self.batch_size, self.number_of_kernel, self.input_size[1], out_shape[0] * out_shape[1]))

        for kernel_index, padded_kernels in enumerate(padded_kernelss):
            for depth_index, padded_kernel in enumerate(padded_kernels):
                toeplitz_list = []

                for i in range(padded_kernel.shape[-2] - 1, -1, -1):  # iterate from last row to the first row
                    # i th row of the F
                    c = padded_kernel[i, :]
                    # first row for the toeplitz fuction should be defined otherwise
                    r = np.r_[c[0], np.zeros(I_col_num - 1)]
                    c = np.asarray(c).ravel()
                    r = np.asarray(r).ravel()
                    # Form a 1D array of values to be used in the matrix, containing a reversed
                    # copy of r[1:], followed by c.
                    vals = np.concatenate((r[-1:0:-1], c))
                    toeplitz_m = vals[indexes]
                    toeplitz_list.append(toeplitz_m)

                doubly_blocked = np.zeros((doubly_blocked_shape[0], doubly_blocked_shape[1]))
                for i in range(doubly_indices.shape[0]):
                    for j in range(doubly_indices.shape[1]):
                        start_i = i * b_h
                        start_j = j * b_w
                        end_i = start_i + b_h
                        end_j = start_j + b_w
                        doubly_blocked[start_i: end_i, start_j:end_j] = toeplitz_list[doubly_indices[i, j] - 1]

                if d_W:
                    for k in range(self.number_of_kernel):
                        result[kernel_index, k, depth_index] = np_matmul(doubly_blocked,
                                                                                   vectorized_image_batch[kernel_index,
                                                                                                          k])
                else:
                    if forward:
                        for batch_index in range(self.batch_size):
                            result[batch_index, kernel_index, depth_index] = np_matmul(doubly_blocked,
                                                                                       vectorized_image_batch[batch_index,
                                                                                                              depth_index])
                    else:
                        for batch_index in range(self.batch_size):
                            result[batch_index, kernel_index, depth_index] = np_matmul(doubly_blocked,
                                                                                       vectorized_image_batch[batch_index,
                                                                                                              kernel_index])

        result = np.reshape(result, (self.batch_size, self.number_of_kernel, self.input_size[1], out_shape[0], out_shape[1]))
        result = result[:, :, :, ::-1, :]

        if I_row_num < F_row_num:
            F_row_num = I_row_num
            F_col_num = I_col_num

        if mode == 'valid':
            result = result[:, :, :, (F_row_num - 1):-(F_row_num - 1), (F_col_num - 1):-(F_col_num - 1)]
        if mode == 'same':
            result = result[:, :, :, (F_row_num - 1) // 2:-((F_row_num - 1) - ((F_row_num - 1) // 2)),
                     (F_col_num - 1) // 2:-((F_col_num - 1) - ((F_col_num - 1) // 2))]

        if forward:
            # sum for depth
            result = np.sum(result, axis=2)
            self.z = result
        else:
            if d_W:
                return result
            else:
                # sum for kernels
                result = np.sum(result, axis=1)
                self.output_bp = result


class Flatten(Layer):
    def __new__(cls):
        def set_prev_layer(layer):
            instance = super(Flatten, cls).__new__(cls)
            instance.__init__(prev_layer=layer)
            return instance
        return set_prev_layer

    def __init__(self, prev_layer=None):
        super().__init__(prev_layer=prev_layer)

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.input_size = self.prev_layer.output_size

        self.set_output_size()

        log = "Flatten layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def set_output_size(self):
        output_size = 1
        for i in self.input_size[1:]:
            output_size *= i
        self.output_size = (self.batch_size, 1, output_size)

    def forward_process(self):
        """Flatten connected layer forward process."""
        x = self.prev_layer.output
        self.output = x.reshape(self.batch_size, 1, -1)

        assert self.output.shape == self.output_size

        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """Flatten connected layer backward process"""
        self.output_bp = input_error.reshape(self.prev_layer.output_size)

        assert self.output_bp.shape == self.input_size
        self.prev_layer.backward_process(self.output_bp)


class Pool2d(Layer):
    def __new__(cls, kernel_size):
        def set_prev_layer(layer):
            instance = super(Pool2d, cls).__new__(cls)
            instance.__init__(prev_layer=layer, kernel_size=kernel_size)
            return instance
        return set_prev_layer

    def __init__(self, kernel_size, prev_layer=None):
        super().__init__(prev_layer=prev_layer)
        self.kernel_size = kernel_size

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        self.batch_size = batch_size
        self.input_size = self.prev_layer.output_size

        self.set_output_size()

        log = "2D pool layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
        print(log)

        for layer in self.next_layer:
            layer.set_size_forward(batch_size, learning_rate, optimizer)

    def set_output_size(self):
        self.output_size = (self.batch_size, self.input_size[1],
                            self.input_size[2] // self.kernel_size, self.input_size[3] // self.kernel_size)

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """2D pool layer forward process."""
        strided = np.lib.stride_tricks.as_strided

        def view_as_blocks(arr_in, block_shape):
            block_shape = np.array(block_shape)
            arr_shape = np.array(arr_in.shape)
            arr_in = np.ascontiguousarray(arr_in)

            new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
            new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

            arr_out = strided(arr_in, shape=new_shape, strides=new_strides)

            return arr_out

        batch_input = np.reshape(self.prev_layer.output[:, :, :(self.input_size[2] - self.input_size[2] % self.kernel_size),
                                 :(self.input_size[3] - self.input_size[3] % self.kernel_size)],
                                 (-1,
                                  self.input_size[2] - (self.input_size[2] % self.kernel_size),
                                  self.input_size[3] - (self.input_size[3] % self.kernel_size)))

        blocks = np.array([view_as_blocks(m, block_shape=(self.kernel_size, self.kernel_size)) for m in batch_input])
        self.output = np.reshape(np.max(blocks, axis=(3, 4)), self.output_size)

        assert self.output.shape == self.output_size
        for layer in self.next_layer:
            layer.forward_process()

    def backward_process(self, input_error):
        """2D pool layer backward process"""

        input_error = np.pad(input_error.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3),
                             ((0, 0), (0, 0), (0, self.input_size[2] % self.kernel_size),
                             (0, self.input_size[3] % self.kernel_size)), 'constant', constant_values=0)
        output_exp = np.pad(self.output.repeat(self.kernel_size, axis=2).repeat(self.kernel_size, axis=3),
                            ((0, 0), (0, 0), (0, self.input_size[2] % self.kernel_size),
                            (0, self.input_size[3] % self.kernel_size)), 'constant', constant_values=0)

        mask = np.equal(self.prev_layer.output, output_exp).astype(int)
        self.output_bp = mask * input_error

        assert self.output_bp.shape == self.input_size
        self.prev_layer.backward_process(self.output_bp)


class Concat(Layer):
    def __new__(cls, axis=1):
        def set_prev_layer(layer):
            """layer: list of the concatenated layers [x1, x2]"""

            instance = super(Concat, cls).__new__(cls)
            instance.__init__(prev_layer=layer, axis=axis)
            return instance
        return set_prev_layer

    def __init__(self, prev_layer=None, axis=1):
        super().__init__(prev_layer=prev_layer)
        self.axis = axis + 1  # because the batch size

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        os0 = self.prev_layer[0].output_size
        os1 = self.prev_layer[1].output_size

        if os0 is not None and os1 is not None:
            self.batch_size = batch_size

            self.set_output_size()
            self.set_input_size()

            log = "Concat layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
            print(log)

            for layer in self.next_layer:
                layer.set_size_forward(batch_size, learning_rate, optimizer)
        else:
            # It takes the process back to a not calculated layer.
            pass

    def prev_layer_set_next_layer(self):

        for layer in self.prev_layer:
            layer.set_next_layer(self)

    def set_input_size(self):
        input_size = []

        for layer in self.prev_layer:
            input_size.append(layer.output_size)

        self.input_size = tuple(input_size)

    def set_output_size(self):
        output_size_at_axis = 0
        output_size = []

        for layer in self.prev_layer:
            output_size_at_axis += layer.output_size[self.axis]

        for i, size in enumerate(self.prev_layer[0].output_size):
            if i == self.axis:
                output_size.append(output_size_at_axis)
            else:
                output_size.append(size)

        self.output_size = tuple(output_size)

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Concatenate layer forward process."""

        x0 = self.prev_layer[0].output
        x1 = self.prev_layer[1].output

        if x0 is not None and x1 is not None:
            try:
                self.output = np.concatenate((x0, x1), axis=self.axis)
            except ValueError:
                print("In layer Concat, the two layers don't have the same dimension at the appropriate axis!")

            assert self.output.shape == self.output_size

            for layer in self.next_layer:
                layer.forward_process()
        else:
            # It takes the process back to a not calculated layer.
            pass

    def backward_process(self, input_error):
        """Concatenate layer backward process"""

        s0 = self.prev_layer[0].output_size[self.axis]
        s1 = self.prev_layer[1].output_size[self.axis]

        self.output_bp = np.split(input_error, [s0], axis=self.axis)

        assert self.output_bp[1].shape[self.axis] == s1

        for i, layer in enumerate(self.prev_layer):
            layer.backward_process(self.output_bp[i])


class Add(Layer):
    def __new__(cls, weights_of_layers=None):
        def set_prev_layer(layer):
            """layer: list of the added layers [x1, x2]"""

            instance = super(Add, cls).__new__(cls)
            instance.__init__(prev_layer=layer, weights_of_layers=None)
            return instance
        return set_prev_layer

    def __init__(self, prev_layer=None, weights_of_layers=None):
        super().__init__(prev_layer=prev_layer)
        # weigths at the addition
        if weights_of_layers:
            self.weights_of_layers = weights_of_layers
        else:
            self.weights_of_layers = [1, 1]

    def set_size_forward(self, batch_size, learning_rate, optimizer):
        os0 = self.prev_layer[0].output_size
        os1 = self.prev_layer[1].output_size

        if os0 is not None and os1 is not None:
            assert os0 == os1
            self.batch_size = batch_size

            self.output_size = os0
            self.input_size = os0

            log = "Add layer with {} parameters.\nInput size: {}\nOutput size: {}\n".format(0, self.input_size, self.output_size)
            print(log)

            for layer in self.next_layer:
                layer.set_size_forward(batch_size, learning_rate, optimizer)
        else:
            # It takes the process back to a not calculated layer.
            pass

    def prev_layer_set_next_layer(self):

        for layer in self.prev_layer:
            layer.set_next_layer(self)

    def save_weights(self, w_array):
        for layer in self.next_layer:
            layer.save_weights(w_array)

    def load_weights(self, w_array):
        for layer in self.next_layer:
            layer.load_weights(w_array)

    def forward_process(self):
        """Add layer forward process."""

        x0 = self.prev_layer[0].output
        x1 = self.prev_layer[1].output

        if x0 is not None and x1 is not None:
            try:
                x0 *= self.weights_of_layers[0]
                x1 *= self.weights_of_layers[1]

                self.output = np.add(x0, x1)
            except ValueError:
                print("In layer Add, the two layers don't have the same shape!")

            assert self.output.shape == self.output_size

            for layer in self.next_layer:
                layer.forward_process()
        else:
            # It takes the process back to a not calculated layer.
            pass

    def backward_process(self, input_error):
        """Add layer backward process"""

        self.output_bp = input_error
        assert self.output_bp.shape == self.input_size

        for i, layer in enumerate(self.prev_layer):
            layer.backward_process(self.output_bp * self.weights_of_layers[i])


