from keras.layers import Layer, Dense, ReLU

# This is a class that represents a feedforward neural network layer with two fully
# connected (dense) layers and a ReLU activation function between them. It has three instance variables:
# fully_connected1: a dense layer with d_ff units.
# fully_connected2: a dense layer with d_model units.
# activation: a ReLU activation layer.
# The call method of this class applies the feedforward layer to an input tensor x.
# It does this by first applying the first fully connected layer to x, then applying the ReLU activation
# function to the result, and finally applying the second fully connected layer to the result
# of the activation function. The output of the second fully connected layer is then returned as
# the output of the feedforward layer.

class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fully_connected1 = Dense(d_ff)  # First fully connected layer
        self.fully_connected2 = Dense(d_model)  # Second fully connected layer
        self.activation = ReLU() # ReLU activation layer

    def call(self, x):
        # The input is passed into the two fully-connected layers, with a
        # ReLU in between
        x_fc1 = self.fully_connected1(x)
        return self.fully_connected2(self.activation(x_fc1))
