from keras.layers import Layer, Dropout
from numpy import random

from EncoderLayer import EncoderLayer

# to be implemented
from PositionEmbeddingFixedWeights import PositionEmbeddingFixedWeights


# The Transformer encoder consists of a stack of N identical layers, where
# each layer further consists of two main sub-layers:
# 1. The first sub-layer comprises a multi-head attention mechanism that
# receives the queries, keys, and values as inputs
# 2. A second sub-layer comprises a fully- connected feed-forward network
# Implementing the Encoder.
# On top of each of these two sub-layers there is a Add&Norm layer.
# The inputs of this sub-layer are the output of the corresponding
# preceding layer and the skip connection. The output of this layer
# is LayerNorm(Sublayer Input)

class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)
        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):
        # Generate the positional encoding
        pos_encoding_output = self.pos_encoding(input_sentence)

        # Add in a dropout layer
        x = self.dropout(pos_encoding_output, training=training)

        # Pass on the positional encoded values to each encoder layer
        for i, layer in enumerate(self.encoder_layer):
            x = layer(x, padding_mask, training)

        return x


# TEST RUN

# These parameters are specified in the paper "Attention Is All You Need"
# by Vaswani et al. (2017)
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_ff = 2048  # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers outputs
n = 6  # Number of layers in the encoder stack

batch_size = 64  # Batch size from the training process
dropout_rate = 0.1  # frequency of dropping the input units in the dropout layers

# For the input sequence, we will work with dummy data for the time being
# until we arrive at the stage of training the complete transformer model
# at which point we will be using the actual sentences.

enc_vocab_size = 20  # Vocabulary size for the encoder
input_seq_length = 5  # Maximum length of the input sequence

input_seq = random.random((batch_size, input_seq_length))

# Next we create a new instance of the Encoder class, assigning its
# output to the encoder variable, subsequentially feeding in the input
# arguments, and printing the result. We will set the padding mask
# argument to None for the time being, but we will retur to this as we
# implement the complete transformer model.

encoder = Encoder(enc_vocab_size, input_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)
print(encoder(input_seq,None,True))
