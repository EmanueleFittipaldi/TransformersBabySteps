from numpy import array, random
from scipy.special import softmax


def scaled_dot_product_attention(queries, keys, values, mask):
    # We now score the query vectors against all the key
    # vectors.
    scores = queries @ keys.transpose()

    # It is common pracice to divide the score
    # values by the square root of the dimensionality of the key vecors (three,
    # in this case) to keep the gradients stable.
    scores = scores / keys.shape[1] ** 0.5
    print("scores:{}\n".format(scores))

    #  For each of these large negative inputs, the softmax function will
    #  in turn, produce an output value that is close to zero, effectively masking them out
    if mask is not None:
        scores += (mask * -1e9)

    # The score values are then passed through a softmax operation to generate
    # the weights.
    weights = softmax(scores)
    print("weights:{}\n".format(weights))

    # We finally compute the attention output as a weighted sum of all four
    # value vectors.
    attention = weights @ values
    print("attention:{}".format(attention))
    return attention


# TEST RUN:
# In this tutorial, we will discover the attention mechanism and its
# implementation. Specifically we will learn:
# - How the attention mechanism uses a weighted sum of all the encoder
#   Hidden states (fancy name for the encoder inputs) to fexibly focus the attention fo the decoder to the
#   most relevant parts of the input sequence
# - How the attention mechanism can be generalized for tasks where the
#   information may not necessarily be related in a sequential fashion.
# - How to implement the general attention mechanism with NumPy and SciPy.

# encoder representations of four different words. In a real scenario, these
# vectors would have been generated by an encoder. We are going to compute
# the attention for each word in this sequence of four words.
word_1 = array([1, 0, 0])
word_2 = array([0, 1, 0])
word_3 = array([1, 1, 0])
word_4 = array([0, 0, 1])

words = array([word_1, word_2, word_3, word_4])
print("Words:{}\n".format(words))

# Next, we generates the weight matrices that we will eventually multiply
# to the word representations to generate the queries, keys and values.
# Here for practical purpose we generate these weight matrices randomly, but
# in a real scenario these would have been learned during training.
random.seed(42)  # to allow us to reproduce the same attention values
W_Q = random.randint(3, size=(3, 3))
print("W_Q:{}".format(W_Q))
W_K = random.randint(3, size=(3, 3))
print("W_K:{}".format(W_K))
W_V = random.randint(3, size=(3, 3))
print("W_V:{}\n".format(W_V))

# Next, we generate the query, key, value vectors for each word, by
# multiplying each word representation by each of the weight matrices.
Q = words @ W_Q
print("Q:{}".format(Q))
K = words @ W_K
print("K:{}".format(K))
V = words @ W_V
print("V:{}\n".format(V))

scaled_dot_product_attention(Q, K, V, None)
