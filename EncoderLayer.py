from keras.layers import Layer, Dropout

# Imprting all the elements that makes up the Transformer Encoder
from MultiHeadAttention import MultiHeadAttention
from AddNorm import AddNormalization
from FeedForward import FeedForward


# Implementing the Encoder Layer. The transformer will replicate this
# layer identically N times.
# A note on dropout: This is called a regularization technique and is
# one of many kind. It is used to avoid overfitting. It takes "rate"
# as a parameter which indicates basically how many input units are
# set to 0 in order to not let them partecipate in the training phase.
# By doing so we will have a more generalized model.
class EncoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    # The padding mask here serves the purpose of obscuring the zeroes
    # in the input sequences from being processed. These zeroes were
    # placed only as padding so they are not useful.
    # The training flag when set to True, will only apply the dropout
    # òayers during training.
    def call(self, x, padding_mask, training):
        # Multi-head attention layer
        multihead_output = self.multihead_attention(x, x, x, padding_mask)
        print("multihead_output shape:{}".format(multihead_output))

        # Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training=training)

        # Followed by an Add & Norm layer
        addnorm_output = self.add_norm1(x, multihead_output)

        # Followed by a fully connected layer
        feedforward_output = self.feed_forward(addnorm_output)

        # Add in another dropout layer
        feedforward_output = self.dropout2(feedforward_output, training=training)

        # Followed by another Add & Norm layer
        return self.add_norm2(addnorm_output, feedforward_output)
