import numpy as np # helps with the math
import matplotlib.pyplot as plt # to plot error during training
import copy


def plot_error(epocs, error_history) :

    plt.plot ( epocs, error_history, label="errors" )

    plt.title ( "Error Rates" )
    plt.xlabel ( "EPOCs" )
    plt.ylabel ( "error" )
    plt.legend ()
    plt.show ()

def plot_weights(epocs, weight_history) :
    wh = np.array ( weight_history )
    w_0 = wh[:, 0]
    w_1 = wh[:, 1]
    w_2 = wh[:, 2]
    plt.plot ( epocs, w_0, label="weight 0" )
    plt.plot ( epocs, w_1, label="weight 1" )
    plt.plot ( epocs, w_2, label="weight 2" )

    plt.title ( "Weights" )
    plt.xlabel ( "EPOCs" )
    plt.ylabel ( "weight" )
    plt.legend ()
    plt.show ()

# input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])

# create NeuralNetwork class
class NeuralNetwork:

    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as normal random vars
        self.weights = np.array([[np.random.normal()], [np.random.normal()], [np.random.normal()]])
        self.error_history = []
        self.epoch_list = []
        self.weight_history = []

    #activation function ==> S(x) = 1/1+e^(-x)
    def sigmoid(self, x, deriv=False):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid_derivative(self.hidden)
        self.weights += np.dot(self.inputs.T, delta)
        self.weight_history.append( copy.deepcopy(self.weights) )

    def train(self, epochs=100):
        for epoch in range(epochs):
            # flow forward and produce an output
            self.feed_forward()
            # go back though the network to make corrections based on the output
            self.backpropagation()    
            # keep track of the error history over each epoch
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()

# create two new examples to predict                                   
example = np.array([[1, 1, 0]])
example_2 = np.array([[0, 1, 1]])

# print the predictions for both examples                                   
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])


plot_error(NN.epoch_list, NN.error_history)
plot_weights(NN.epoch_list, NN.weight_history)
