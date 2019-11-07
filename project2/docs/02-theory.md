# Theory

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


### Feed-forward

The basic idea of this kind of network is that the input information (i. e. the
features/predictors of our data set) $y$, are passed through all of the neurons in
the hidden layers, until it ends up in the output layer. Figure @fig:nn shows
an example with three inputs, two hidden layers with four neurons each, and one
output. This process is called *forward feeding* of the network. Each input is
multiplied by a weight $w_i$ when passed into a neuron, and the result is
called $z_i$. The signal is then evaluated by an activation function $a(z_i) =
a_i$, which then becomes the output from each neuron. Since we are dealing with
several observations, we assemble the weights $w_i$ into a matrix $W$. Usually
we add a bias $b$ to each of the neurons in a hidden layer, to prevent outputs
of only zeros. The output $a_l$ of a layer $l$ then becomes (where $a_{l-1}$ is the
output from the previous layer):

$$a_l = a(a_{l-1} W_l + b_l)$${#eq:feedforward}

In the case of the first hidden layer, the input matrix $X$ will take the place
of $a_{l-1}$. The weights are usually initialized with random
values, for example using a normal or uniform distribution, because if all the
weights were the same, all neurons would give the same output. In this project
we use a standard normal distribution when doing classification, but in the
case for regression we use an initialization proposed by Xavier Glorot and
Yoshua Bengio[@glorot_understanding_nodate], where we scale the
randomly distributed weights by a factor $1/\sqrt{n_{l-1} + n}$ ($n_{l-1}$ is
the size of the previous layer, and $n_l$ is the size of the current layer).
The biases are given a small non-zero value, in our case $b_i = 0.01$ for all
layers, in order ot ensure activation.

![Schematic neural network. The circles represent neurons, and the colors
indicate what layer they are a part of: Grey means input layer, white means
hidden layer and black means output layer. The arrows show that all neurons of
one layer is connected to each of the neurons in the next layer; i. e. we have
only fully connected layers.](figs/neural-network.png){#fig:nn}



### Activation functions

The choice of activation function may have a huge effect on the model, and
there are several options that may work depending on what data set we want to
process, how it is scaled and if it is a regression or classification case. In
a feed-forward neural network, we must have activation functions that are
non-constant, bounded, monotonically-increasing and continous for the network
to function correctly on complex data sets. It is possible to choose different
activation functions for each hidden layer, but in this project we have the
same activation function throughout the networks we use.

Common choices for activation functions are the logistic/sigmoid function, the
Rectified Linear Unit function (ReLU) and the $\tanh$-function. For
classification problems, the sigmoid function, presented in equation
@eq:sigmoid, is often preferred. This function gives only output that is
between $0$ and $1$, which is good when we want to predict the probability as
output of the network. In multiclass classification problems the so-called
softmax function is a better choice, because it forces the sum of probabilites
for the possible classes to be 1, but for binary classification problems, like
we have in this project, the sigmoid function is sufficient.

For regression we use another common activation function, the ReLU function:

$$\text{ReLU} = \begin{cases}
    0 & x < 0 \\
    x & x \geq 0
\end{cases}
$$


### Output layer and back propagation

After the feed-forward process is done, the output layer will contain the
predictions of the neural network, which we will compare with the true targets
of our training data set. Based in this comparison, we will adjust the weights
and biases of the network in order to improve the performance. Since the
weights and biases usually are initialized randomly, the first feed-forward
iteration will most likely give very wrong results. One feedforward pass is
usually called an *epoch*, and we choose the number of epochs based on how much
"training" the network needs in order to give satisfactory results. The process
of adjusting the weights and biases is called *back propagation*.

The comparison between the output from the network $\hat{y}$ and the true
targets of our training data set $y$ is done with a predefined cost
function $\mathcal{C}$. We want to minimize the
cost, and to do this we need to know how the weights and biases must be
adjusted to obtain a better result. This is done by calculating the gradient of
the cost function, and the exact derivation depends on how the cost function is
defined. In general terms the expression for the output error $\delta_L$ becomes

$$\delta_L = \frac{\partial \mathcal{C}}{\partial a_L} \frac{\partial
a_L}{\partial {z_L}},$$

and the back propagate error for the other layers $l = L-1, L-2, ..., 2$ is

$$\delta_l = \delta_{l+1} W_{l+1}^T a'(z_l)$$,

where $a'$ is the derivative of the activation function. The update of the weights and biases for a general layer $l$ is calculated by

$$W_l \leftarrow = W_l - \eta a_{l-1}^T \delta_l,$$
$$b_l \leftarrow = b_l - \eta \delta_l,$$

where $\eta$ is the learning rate, which specifies how much the weights and
biases should be adjusted for each back propagation.

In our classification problem, we use the binary cross-entropy as a cost
function:

$$\mathcal{C}_{\text{classification}} = - (y \log (p) + (1-y) \log (1 - p)),$$

where $p$ is the predicted probability of an observation, and $y$ is the true
target, which will be either $0$ or $1$ since we have a binary classification
problem. The probabilities $p$ is the same as the output for our network
$\hat{y}$. Combining this with the sigmoid as the activation function in the
last layer, we get an output error

$$\delta_L = \hat{y} - y.$${#eq:outputerror}

For regression we use a slightly modified version of the mean squared error:

$$\mathcal{C} = \frac{1}{2} \sum_{i=1}^n (\hat{y}_i - y_i)^2$$

The factor $\frac{1}{2}$ is used to simplify the derivative of the
cost function, because if we use the identity function (which returns the same
values that is the input to the function) as an activation function in the last
layer, the output error is the same as in equation @eq:outputerror, which is
very convenient because we can use the same calculation whether we are doing
classification or regression. A common practice when using neural networks on
regression problems is to use the ReLU as activation function in the hidden
layers, and then use the identity function in the last layer.

<!-- \footnote{This is the default behaviour of Scikit-Learn's implementation of
\texttt{MLPRegressor}, which trains a neural network for regression cases.} -->


