# Theory

## Logistic regression




### Gradient descent




## Artificial neural networks

*Artificial neural networks* (ANN) is a machine learning technique inspired by
the networks of neurons that make up the biological brain. There are several
different types of neural networks, and in this project we use a
multi-layer perceptron (MLP), where there is one or more *hidden layers* between
the input and output layers, and each hidden layer consists of several
*neurons*. The signal will only go in one direction, from the input layer to
the output layer, which makes this a *feedforward neural network* (FNN).
Additionally, all neurons of a layer will be connected to every neuron on the
previous layer, which makes it a *fully connected layer*. In this section we
will review the essential parts of how a neural network functions.


### Information flow and forward feeding

The basic idea of this kind of network is that the input information (i. e. the
features/predictors of our data set), are passed through all of the neurons in
the hidden layers, until it ends up in the output layer. Figure @fig:nn shows
an example with three inputs, two hidden layers with four neurons each, and one
output. This process is called *forward feeding* of the network. Each input is
multiplied by a weight $w_i$ when passed into a neuron, and the result is
called $z_i$. The signal is then evaluated by an activation function $a(z_i) =
a_i$, which then becomes the output from each neuron. Since we are dealing with
several observations

![Schematic neural network. The circles represent neurons, and the colors
indicate what layer they are a part of: Grey means input layer, white means
hidden layer and black means output layer. The arrows show that all neurons of
one layer is connected to each of the neurons in the next layer; i. e. we have
only fully connected layers.](figs/neural-network.png){#fig:nn}

Usually we add a bias to each of the neurons in a hidden layer, to prevent
outputs of only zeros.


### Activation functions

The choice of activation function may have a huge effect on the model, and
there are several options that may work depending on what data set we want to
process, how it is scaled and if it is a regression or classification case.
Common choices are the logistic/sigmoid function, the Rectified Linear Unit
function (ReLU) and the $\tanh$-function. In our project we have chosen to use
primarily the sigmoid function, presented in equation @eq:sigmoid. This
function gives only output that is between $0$ and $1$.


### Back propagation




### Minibatches
