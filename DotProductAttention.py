from keras.layers import Layer
from numpy import random
from tensorflow import matmul, math, cast, float32
from keras.backend import softmax


# Implementing the Scaled-Dot Product Attention using NumPy and Keras.
# This class extends the Layer class from Keras. The "Layer" class is the
# base class for all the layers in a model. It provides common functions for
# working with layers.
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scores = (Q * K^T) / sqrt(d_k)
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores. The mask will contain either 0
        # values to indicate that the corresponding token in the input
        # sequence should be considered in the computations or a 1 to indicate
        # otherwise. The mask will be multiplied by -1e9 (-1*10^9) to set the 1
        # values to a large negative numbers.
        if mask is not None:
            scores += -1e9 * mask

        # Applying softmax to weights
        weights = softmax(scores)

        # Attention Matrix = weights * values
        return matmul(weights, values)


# Testing Out the code
# Here we are working with the parameter values as specified in the paper
# "Attention" Is All You Need" by Vaswani et al. (2017)


#d_k = 64  # 64 Dimensionality of the linearly projected queries and keys
#d_v = 64  # 64 Dimensionality of the linearly projected values
#batch_size = 64 # 64 How many "packets" of #input_seq_length to process
#input_seq_length = 5  # 5 Maximum length of the input sequence


# Until we will actual train the complete transformer model, we are going to
# use dummy data. In the complete Transformer model, values for the sequence
# length and the queries, keys and values will be obtained through a process of
# word tokenization and embedding.
#queries = random.random((batch_size, input_seq_length, d_k))
#keys = random.random((batch_size, input_seq_length, d_k))
#values = random.random((batch_size, input_seq_length, d_v))

# We now create an istance of the DotProductAttention class, assigning its output
# to the attention variable.
#attention = DotProductAttention()

# Running this code, produces an output of shape(batch size, sequence length, values
# dimensionality).
#print(attention(queries, keys, values, d_k))
