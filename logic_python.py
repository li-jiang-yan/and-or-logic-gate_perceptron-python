from math import exp

# Starting weights
w1 = 0.3
w2 = -0.7
b = 0.1
training_rate = 20

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + exp(-x))

# Each row: input 1, input 2, target output
lookup_table = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

# Column header
print(" E | S | i1 | i2 |   w1 |   w2 |    b | Ot | Or |  err | Delta_w1 | Delta_w2 |  Delta_b")

for e in range(8):
    for s in range(4):
        i1 = lookup_table[s][0] # input 1
        i2 = lookup_table[s][1] # input 2
        Ot = i1 and i2          # target output
        Or = sigmoid(i1 * w1 + i2 * w2 + b)
        err = Ot - Or
        Delta_w1 = training_rate * Or * err * (1 - Or) * i1
        Delta_w2 = training_rate * Or * err * (1 - Or) * i2
        Delta_b = training_rate * Or * err * (1 - Or)
        print("{:>2} | {} |  {} |  {} | {: .1f} | {: .1f} | {: .1f} | {: .0f} | {: .0f} | {: .1f} | {: .5f} | {: .5f} | {: .5f}".format(e, s, i1, i2, w1, w2, b, Ot, Or, err, Delta_w1, Delta_w2, Delta_b))
        w1 += Delta_w1
        w2 += Delta_w2
        b += Delta_b

print("Final solution: w1 = {}, w2 = {}, b = {}".format(w1, w2, b))
