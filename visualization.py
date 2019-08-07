import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def log(x):
    return 1 / (1 + np.exp(-1 * x))


with open(os.path.join('weights_256_MSE.txt'), 'rb') as fp:
    weights = pickle.load(fp)

W = []
b = []

for i in range(len(weights) // 2):
    W.append(weights[i * 2])
    b.append(weights[i * 2 + 1])

digits_hotmap = []

for j, w in enumerate(W):
    digits = []
    print(w.shape)
    for i in range(w.shape[1]):
        digit = []
        if j == 0:
            digit = w[:, i]
            digit = np.reshape(digit, (28, 28))
            # plt.imshow(digit)
            # plt.show()
            # digit = log(digit)
            digits.append(digit)
        else:
            for k in range(w.shape[0]):
                if k == 0:
                    digit = digits_hotmap[-1][k] * w[k, i]
                else:
                    digit += digits_hotmap[-1][k] * w[k, i]

            digits.append(digit)
    digits_hotmap.append(digits)

for i in range(len(digits_hotmap[-1])):
    max_d = np.amax(digits_hotmap[-1][i])
    min_d = np.amin(digits_hotmap[-1][i])
    digits_hotmap[-1][i] = (digits_hotmap[-1][i] - min_d) / (max_d - min_d)

print(np.array(digits_hotmap[-1]).shape)

for i, digit in enumerate(digits_hotmap[-1]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digit)
    plt.title(i)

plt.show()


