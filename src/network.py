import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Structure of inputs -> processing -> output
        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 1

        # Defining weights and bias with rando values
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        print('first weight')
        print(self.w1)
        self.b1 = np.zeros((1, self.hidden_size))
        print('first bias')
        print(self.b1)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)
        print('second weight')
        print(self.w2)
        self.b2 = np.zeros((1, self.output_size))
        print('second bias')
        print(self.b1)
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, x, y, output, learning_rate):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        error_hidden = d_output @ self.w2.T
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        self.w2 += self.a1.T @ d_output * learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate

        self.w1 += x.T @ d_hidden * learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate
