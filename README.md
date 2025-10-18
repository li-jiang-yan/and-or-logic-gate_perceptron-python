# ann-and-or-logic-gate-training
This endeavors to implement the training of an AND/OR logic gate using artificial neural networks

# Optimization Theory
We want to minimize

$$E(O_{r})=\frac{1}{2}[O_{t}-O_{r}]^2$$

where

$$O_{r}(w_{1},w_{2},b)=\sigma(w_{1}i_{1}+w_{2}i_{2}+b)$$

Gradient descent to minimize $E$ can be done with

$$\Delta w_{j}=-\eta\frac{\partial E}{\partial w_{j}}$$

(here $b$ can be thought of another $w$ where $i$ is always 1)

$$\Delta w_{j}=-\eta\frac{\partial E}{\partial w_{j}}=\eta O_{r}(O_{t}-O_{r})(1-O_{r})i_{j}$$
