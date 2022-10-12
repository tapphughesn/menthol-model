import numpy as np
from glob import glob
from matplotlib import pyplot as plt

fs = glob("/Users/nick/Documents/Gillings_work/outputs/numpy_arrays/*2022-10-12*")
print(len(fs))

x = np.arange(0,51,1)
ind = 1

for f in fs:
    arr = np.load(f)
    arr = np.sum(arr, axis=(1,2))
    arr = arr[:,ind] / np.sum(arr[:,:-1], axis=1)
    plt.plot(x, arr)

plt.show()