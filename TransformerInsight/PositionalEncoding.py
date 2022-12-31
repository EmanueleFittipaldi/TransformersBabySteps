import numpy as np


# import matplotlib.pyplot as plt

# Positional Encoding and why we need it:
# LSTM take word embeddings sequentially,which is why they're so slow.
# A pro of working that way Is that they know which word came first which word came second and so on.
# Transformers on the other hand take up all embeddings at once. Even though this is a huge plus and makes
# transformers much faster, the downside is that they lose the critical information related to word ordering.
# In simple words they are not aware of which word came first in the sequence and which came last. This is a problem
# because position information matters.
# In order to bring back the word order information back to transformers without having to make them recurrent like
# LSTM we introduce a new set of vectors containing the position information. We call them the position embeddings.
# We can start by simply adding the word embeddings to their corresponding position embeddings and create new order
# aware word embeddings. But what values should our position embeddings contain?
# Ideally the position embedding values at a given position should remain the sameirrespective of the text total length
# or any other factor. We use wave frequencies to capture position information.
# Example: Let us take the first position embedding as an example therefore the pos variable in the formula will
# be 0. Next the size of the position embedding has to be the same as the word embeddings and so it is set to be 512
# as in the original paper "Attention Is All You Need" of Vaswani et al. (2017).
# This is represented by the letter d in our formula. The letter i represents the indices of each of the position
# embedding dimensions.
# Now if we plot a sinusoidal curve by varying the variable i indicating word positions on the
# x-axis we will get a smooth looking curve.
# Now since the height of the sine curve depends on the position on the
# x-axis we can very well use the curve's height as a proxy to work positions since the curve height only varies
# between a fixed range and is not dependent on the input text length.
# If you plot the curve at different values of i's we get a series of curves of different frequencies.
# Now here is the idea behind having curves of different frequencies: If two points are close by on the curve
# they will remain identical at higher frequencies too it is only at much higher frequencies that their y-coordinates
# on the curve differ and you may be able to tell them apart. For points further apart on the other hand you should be
# able to start seeing them fall on different curve heights quite early on, therefore both the position as well as the
# embedding dimension can inform us of the word order.
# The authors did not use only the sine curves, they use a combination of sine and cosine formula. They used
# an alternative combination of sine and cosine curves that is at odd positions they use the sine formula to get
# their position embeddings and at even positions they use the cosine formula.
def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P

# Code that generated the image "PositionalEncodingFrequencies.png"
# P = getPositionEncoding(seq_len=100, d=512, n=10000)
# plt.imshow(P)
# plt.show()
