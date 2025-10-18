from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import numpy as np

# Inputs and outputs
A = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
b = np.array([0, 1, 1, 1])

# Creating the model
# Use learning_rate = 20.0 or not it is too slow
x = Input(shape=(2,))
y = Dense(1, activation="sigmoid")(x)
model = Model(x, y)
optimizer = Adam(learning_rate=20.0)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Fitting the model
model.fit(A, b, epochs = 30)

# Getting model weights
w_array = model.layers[1].get_weights()[0]
print("[w1, w2] = [{:.2f}, {:.2f}]".format(w_array[0][0], w_array[1][0]))
print("b = {:.2f}".format(float(model.layers[1].get_weights()[1][0])), "\n")

# Checking model predictions
b_pred = np.round(model.predict(A))
print(classification_report(b, b_pred))
