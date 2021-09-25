import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime


def plot_error(epocs, error_history):
    plt.plot(epocs, error_history, label="errors")
    plt.title("Training")
    plt.xlabel("EPOCs")
    plt.ylabel("error")
    plt.legend()
    plt.show()

def plot_weights(epocs, weight_history):
    wh = np.array(weight_history)
    s = wh.shape[1]

    [plt.plot(epocs, wh[:, i], label="weight {0}".format(i) ) for i in range(s)]

    plt.title("Weights")
    plt.xlabel("EPOCs")
    plt.ylabel("weight")
    plt.legend()
    plt.show()

def sigmoidA(x):
    return 1 / (1 + np.exp(-x))

def sigmoidA_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self):

        self.error_history = []
        self.epoch_list = []
        self.weight_history = []
        self.stop_delta = 0.001

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, training_input, sigmoid_fn):
        self.hidden = sigmoid_fn(np.dot(training_input, self.weights))

    def backpropagation(self, training_input, training_output, sigmoid_fn_derivative):
        self.error = training_output - self.hidden
        delta = self.error * sigmoid_fn_derivative(self.hidden)
        self.weights += np.dot(training_input.T, delta)
        self.weight_history.append(copy.deepcopy(self.weights))

    def train(self, training_input, training_output, epochs=250):
        self.weights = np.array( np.random.normal(size=(col, 1)) )
        for epoch in range(epochs):
            stop = self.run_epoch(training_input, training_output, epoch)
            if stop:
                break

    def run_epoch(self, training_input, training_output, epoch):
        stop = False
        self.feed_forward(training_input, sigmoid_fn=sigmoidA)
        self.backpropagation(training_input, training_output, sigmoid_fn_derivative=sigmoidA_derivative)

        err = np.average(np.abs(self.error))
        if err < self.stop_delta:
            stop = True
        self.error_history.append(err)
        self.epoch_list.append(epoch)
        return stop

    def predict(self, sigmoid_fn, new_input):
        prediction = sigmoid_fn(np.dot(new_input, self.weights))
        return prediction

def test_net(nnet):
    run_test_1 = np.array([[1, 1, 1, 0, 0, 1, 0,1]])
    run_test_2 = np.array([[0, 0, 0, 1, 0, 1, 1,0]])
    print('Got: ', nnet.predict(sigmoid_fn=sigmoidA, new_input=run_test_1), ' Expect: ', run_test_1[0][1])
    print('Got: ', nnet.predict(sigmoid_fn=sigmoidA, new_input=run_test_2), ' Expect: ', run_test_2[0][1])

np.random.seed(datetime.now().microsecond)
row, col = 60, 8

inputsA = np.random.randint(2, size=(row,col))
for n in inputsA:
    n[0] = n[col-1]
outputsA = np.array([ inputsA[:, 0] ]).T

NNN = NeuralNetwork()
NNN.train(inputsA, outputsA)

plot_error(NNN.epoch_list, NNN.error_history)
plot_weights(NNN.epoch_list, NNN.weight_history)

test_net(NNN)