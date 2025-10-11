# Finding the optimal training rate so we don't have to run that many epochs (max 30)
from math import exp
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + exp(-x))

# And function
def get_and(a, b):
    return a and b

# Or function
def get_or(a, b):
    return a or b

# Dictionary
function_to_label = {get_and: "and", get_or: "or"}

# Each row: input 1, input 2, target output
lookup_table = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Iterating over all training rates
training_rate_list = np.arange(0.1, 100, 0.1)
epoch = 30

for f in [get_and, get_or]:
    err_final_list = []
    for training_rate in training_rate_list:
        w1 = 0.3
        w2 = -0.7
        b = 0.1
        err_final = 0
        for e in range(epoch):
            for s in range(4):
                i1 = lookup_table[s][0] # input 1
                i2 = lookup_table[s][1] # input 2
                Ot = f(i1, i2)          # target output
                Or = sigmoid(i1 * w1 + i2 * w2 + b)
                err = Ot - Or
                Delta_w1 = training_rate * Or * err * (1 - Or) * i1
                Delta_w2 = training_rate * Or * err * (1 - Or) * i2
                Delta_b = training_rate * Or * err * (1 - Or)
                w1 += Delta_w1
                w2 += Delta_w2
                b += Delta_b
                if e == epoch - 1:
                    err_final += abs(err)
        err_final_list.append(err_final)
    label = function_to_label[f]
    plt.plot(training_rate_list, err_final_list, "-x", label=label)
plt.xlabel("Training rate")
plt.ylabel("Total absolute error")
plt.legend()
plt.show()
