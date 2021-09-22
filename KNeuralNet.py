import numpy as np  # helps with the math
import matplotlib.pyplot as plt  # to plot error during training
import copy


def plot_error(epocs, error_history):
    plt.plot(epocs, error_history, label="errors")
    plt.title("Error Rates")
    plt.xlabel("EPOCs")
    plt.ylabel("error")
    plt.legend()
    plt.show()

def plot_weights(epocs, weight_history):
    wh = np.array(weight_history)
    s = wh.shape[1]
    for i in range(s):
        plt.plot(epocs, wh[:, i], label="weight {0}".format(i))

    plt.title("Weights")
    plt.xlabel("EPOCs")
    plt.ylabel("weight")
    plt.legend()
    plt.show()

inputsA = np.array([[0, 1, 1, 1],
                   [1, 1, 0, 0],
                   [1, 0, 0, 1],
                   [0, 0, 1, 1],
                   [0, 1, 1, 0],
                   [1, 0, 0, 1],
                   [0, 0, 1, 0],
                   [1, 1, 0, 0],
                   [1, 0, 0, 1],
                   [0, 0, 1, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 0]])

outputsA = np.array([inputsA[:, 2]]).T

def sigmoidA(x):
    return 1 / (1 + np.exp(-x))

def sigmoidA_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, seed):
        np.random.seed(seed)

        self.weights = np.array([[np.random.normal()],
                                 [np.random.normal()],
                                 [np.random.normal()],
                                 [np.random.normal()]])
        self.error_history = []
        self.epoch_list = []
        self.weight_history = []
        self.stop_delta = 0.001
        self.epoch_num = 0

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
    run_test_1 = np.array([[0, 1, 1, 0]])
    run_test_2 = np.array([[1, 1, 0, 1]])
    # print the predictions for both examples
    print(nnet.predict(sigmoid_fn=sigmoidA, new_input=run_test_1), ' - Answer: ', 1)
    print(nnet.predict(sigmoid_fn=sigmoidA, new_input=run_test_2), ' - Answer: ', 0)

def flip():
    indices_one = outputsA == 1
    indices_zero = outputsA == 0
    outputsA[indices_one] = 0  # replacing 1s with 0s
    outputsA[indices_zero] = 1  # replacing 0s with 1s

NNN = NeuralNetwork(99)
NNN.train(inputsA, outputsA)

plot_error(NNN.epoch_list, NNN.error_history)
plot_weights(NNN.epoch_list, NNN.weight_history)

test_net(NNN)