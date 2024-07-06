import math
import numpy as np


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# checking
# print(sigmoid(18))

# for handling vector and array stuff like that numpy is good tool


def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))



