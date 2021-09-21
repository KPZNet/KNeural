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
    w_0 = wh[:, 0]
    w_1 = wh[:, 1]
    w_2 = wh[:, 2]
    w_3 = wh[:, 3]
    plt.plot(epocs, w_0, label="weight 0")
    plt.plot(epocs, w_1, label="weight 1")
    plt.plot(epocs, w_2, label="weight 2")
    plt.plot(epocs, w_3, label="weight 3")

    plt.title("Weights")
    plt.xlabel("EPOCs")
    plt.ylabel("weight")
    plt.legend()
    plt.show()


inputs = np.array([[0, 1, 1, 1],
                   [1, 1, 1, 0],
                   [0, 0, 1, 1],
                   [1, 0, 1, 1],
                   [1, 1, 1, 0],
                   [1, 0, 1, 1],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 1],
                   [0, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 0, 0, 1]])

outputs = np.array([inputs[:, 2]]).T


def sigmoidA(x):
    return 1 / (1 + np.exp(-x))

def sigmoidA_derivative(x):
    return x * (1 - x)


class NeuralNetwork:

    def __init__(self, inputs, outputs):
        np.random.seed(99)
        self.inputs = inputs
        self.outputs = outputs
        # initialize weights as normal random vars
        self.weights = np.array([[np.random.normal()],
                                 [np.random.normal()],
                                 [np.random.normal()],
                                 [np.random.normal()]])
        self.error_history = []
        self.epoch_list = []
        self.weight_history = []
        self.stop_delta = 0.01

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, sigmoid_fn):
        self.hidden = sigmoid_fn(np.dot(self.inputs, self.weights))

    def backpropagation(self, sigmoid_fn_derivative):
        self.error = self.outputs - self.hidden
        delta = self.error * sigmoid_fn_derivative(self.hidden)
        self.weights += np.dot(self.inputs.T, delta)
        self.weight_history.append(copy.deepcopy(self.weights))

    def train(self, epochs=2500):
        for epoch in range(epochs):
            stop = self.run_epoch(epoch)
            if stop:
                break

    def run_epoch(self, epoch):
        stop = False
        self.feed_forward(sigmoid_fn=sigmoidA)
        self.backpropagation(sigmoid_fn_derivative=sigmoidA_derivative)

        err = np.average(np.abs(self.error))
        if err < self.stop_delta:
            stop = True
        self.error_history.append(err)
        self.epoch_list.append(epoch)
        return stop

    # function to predict output on new and unseen input data                               
    def predict(self, sigmoid_fn, new_input):
        prediction = sigmoid_fn(np.dot(new_input, self.weights))
        return prediction


# create neural network
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

# create two new examples to predict                                   
example = np.array([[1, 1, 0, 1]])
example_2 = np.array([[0, 1, 1, 1]])

# print the predictions for both examples                                   
print(NN.predict(sigmoid_fn=sigmoidA, new_input=example), ' - Correct: ', 0)
print(NN.predict(sigmoid_fn=sigmoidA, new_input=example_2), ' - Correct: ', 1)

plot_error(NN.epoch_list, NN.error_history)
plot_weights(NN.epoch_list, NN.weight_history)
