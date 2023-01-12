from keras.layers import Layer, Dense
from numpy import random
from tensorflow import reshape, shape, transpose
from DotProductAttention import DotProductAttention

# Multi-head self attention is exactly the same process for computing
# DotProductAttention but repeated h times.


# Theory:
# Each multi-head attention block is made up of four consecutive levels:
# 1. On the first level, three linear (dense) layers that each receive the queries
# keys or values.
# 2. On the second level, a scaled dot-product attention function. The operations
# performed on both the first and second levels are repeated h times and performed
# in parallel, according to the number of heads h composing the multi-head attention
# block.
# 3. On the third level, a concatenation operation that joins the outputs of the
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

    # Next, we will be reshaping the linearly projected(which means that they are the result of the product of W_Q @
    # Words, W_K @ Words and W_V @ Words) queries, keys and values in such a manner as to allow the attention heads
    # to be computed in parallel. For example: if Q,K and V are (number of words x dimension of queries or dimension of
    # keys or dimension of values), we want to split each of these matrix in h matrix of size (number of words x(
    # (dimension of queries or dimension of keys or dimension of values)/h)), that way i have Q_1,...,Q_h;K_1,...,
    # K_h and V_1,...,V_h and I can proceed to compute head_1,...,head_h

    # The outputs of each head_i is of size (number of words x(dimension of queries or dimension of keys or dimension
    # of values)/h) as well. We need to concatenate these outputs in order to pass them to the last Dense layer. The
    # concatenation output is of size (dimension of queries or dimension of keys or dimension of values)/h)
    # concatenated h times. We are basically appending the outputs of each head_i. This concatenation represents the
    # final output of a single scaled dot-product attention process. Since we have repeated this process h times the
    # final multi-head self attention will be of size (dimension of queries or dimension of keys or dimension of
    # values)/h) concatenated h times) concatenated h times

    # See the test run for an example. I've added some prints in order to follow the explaination along with the
    # example.

    # For the purpose of the splitting of Q,K and V, we create
    # a class method whose reshape_tensor.
    # This method receives the matricies Q,K, and V as input
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
        #print("queries reshaped shape:{}".format(q_reshaped.shape))

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        #print("keys reshaped shape:{}".format(k_reshaped.shape))

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        #print("values reshaped shape:{}".format(v_reshaped.shape))

        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        #print("output multi-head attention shape:{}".format(o_reshaped.shape))

        # Once we have generated the multi-head attention output from all the attention
        # heads, the final steps are to concatenate back all outputs together and
        # passing the result through one final dense layer.
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        #print("Concatenation shape:{}".format(output.shape))
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_embedding)
        return self.W_o(output)


# Testing the code
# Here we are working with the parameter values as specified in the paper
# "Attention" Is All You Need" by Vaswani et al. (2017)

#T = 4  # 5 Maximum length of the input sequence
#d_keys = 64  # 64 Dimensionality of the linearly projected queries and keys
#d_values = 64  # 64 Dimensionality of the linearly projected values
#d_embedding = 512  # Dimension of the word embeddings
#h = 8  # Number of self-attention heads
#batch_size = 1  # 64 How many different sentences to compute

#print("T, length of the input sequence:{}\n"
#      "d_keys, shape of the keys K matrix:{}\n"
#      "d_values, shape of the values V matrix:{}\n"
#      "d_embedding, shape of the word embedding:{}\n"
#      "h, number of heads:{}\n"
#      "batch_size, how many sentence to compute simultaneouslt:{}".format(T, d_keys, d_values, d_embedding, h,
#                                                                          batch_size))

# Until we will actual train the complete transformer model, we are going to
# use dummy data. In the complete Transformer model, values for the sequence
# length and the queries, keys and values will be obtained through a process of
# word tokenization and embedding.

#queries = random.random((batch_size, T, d_keys))
#print("Queries shape:{}".format(queries.shape))
#keys = random.random((batch_size, T, d_keys))
#print("Keys shape:{}".format(keys.shape))
#values = random.random((batch_size, T, d_values))
#print("values shape:{}".format(values.shape))

#multihead_attention = MultiHeadAttention(h, d_keys, d_values, d_embedding)
#print("Multi-head attention shape:{}".format(multihead_attention(queries, keys, values).shape))
