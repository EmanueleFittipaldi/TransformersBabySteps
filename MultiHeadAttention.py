from keras.layers import Layer, Dense
from numpy import random
from keras.backend import softmax
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from DotProductAttention import DotProductAttention

# Theory:
# Each multi-head attention block is made up of four consecutive levels:
# 1. On the first level, three linear (dense) layers that each receive the queries
# keys or values.
# 2. On the second level, a scaled dot-product attention function. The operations
# performed on both the first and second levels are repeated h times and performed
# in parallel, according to the number of heads composing the multi-head attention
# block.
# 3. On the third leve, a concatenation operation that joins the outputs of the
# different heads
# 4. On the fourth level, a final linear (dense) layer that produces the output.



# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(d_k)  # Learned projection matrix for the queries
        self.W_k = Dense(d_k)  # Learned projection matrix for the keys
        self.W_v = Dense(d_v)  # Learned projection matrix for the values
        self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output

    # Next, we will be reshaping the linearly projected queries, keys and values in
    # such a manner as to allow the attention heads to be computed in parallel.
    # The queries, keys and values will be fed as input into the multi-head attention
    # block having a shape of (batch size, sequence length, model dimensionality),
    # where the batch size is a hyperparameter of the training process, the
    # sequence length defines the maximum length of the input/output phrases, and the
    # model dimensionality is the dimensionality of the outputs produced by all sub-
    # layers of the mode. They are then passed through the respective dense layer
    # to be linearly projected to a shape of
    # (batch size, sequence length, queries/keys/values dimensionality)
    #
    # The linearly projected queries, keys and values will be rearranged into
    # (batch size, number of heads, sequence length, depth), by first reshaping
    # them into (batch size, sequence, length, number of heads, depth) and then
    # transposing the second and third dimensions. For this purpose, we will create
    # a class method, reshape_tensor.
    #
    # This method receives the linearly projected queries, keys, or values as input
    # (while setting the flag to True) to be rearranged as previously explained. Once
    # the multi-head attention output has been generated, this is also fed into the
    # same function (this time setting the flag to False) to perform a reverse operation,
    # effectively concatenating the result of all heads together.

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    # The next step is to take the reshaped queries, keys and values and feed them
    # into the scaled dot-product attention function. We do it in the call method.
    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Once we have generated the multi-head attention output from all the attention
        # heads, the final steps are to concatenate back all outputs together into a
        # tensor of shape (batch size, sequence length, values dimensionality) and
        # passing the result through one final dense layer.
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)


# Testing the code
# Here we are working with the parameter values as specified in the paper
# "Attention" Is All You Need" by Vaswani et al. (2017)
d_k = 64  # 64 Dimensionality of the linearly projected queries and keys
d_v = 64  # 64 Dimensionality of the linearly projected values
batch_size = 64 # 64 How many "packets" of #input_seq_length to process
input_seq_length = 5  # 5 Maximum length of the input sequence
d_model = 512 # Dimensionality of the model sub-layers outputs
h = 8 # Number of self-attention heads

# Until we will actual train the complete transformer model, we are going to
# use dummy data. In the complete Transformer model, values for the sequence
# length and the queries, keys and values will be obtained through a process of
# word tokenization and embedding.
queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
print(multihead_attention(queries, keys, values))
