# import cudamat as cm
# from cudamat import CUDAMatrix as cmarray
import numpy as np
import time

# cm.cuda_set_device(0)
# np.random.seed(int(time.time()))
#
# k = np.random.rand(1, 3, 3)
# m = np.random.rand(1, 5, 5)
#
# print(k.shape)
# print(m.shape)
#
# cmk = cm.CUDAMatrix(k)
# cmm = cm.CUDAMatrix(m)
#
# print(cmk.shape)
# print(cmm.shape)

# r = cm.correlate(mat=cmm, kernel=cmk)
# print(r.shape)

# ip = list(range(1, 10))

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
x = np.array([[1, 2, 3]])
y = np.array([[1, 2, 3]])
print(np.concatenate((x, y), axis=0))


