import tensorflow as tf
from tensorflow import convert_to_tensor, string
from keras.layers import TextVectorization, Embedding, Layer
import numpy as np



class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)
        self.word_embedding_layer = Embedding(input_dim=vocab_size, output_dim=output_dim,
                                              weights=[word_embedding_matrix], trainable=False)

        self.position_embedding_layer = Embedding(input_dim=sequence_length, output_dim=output_dim,
                                                  weights=[position_embedding_matrix], trainable=False)


    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def call(self, inputs):

        # if a sentence has a length of 5 words the indices will go from 0 to to 4
        position_indices = tf.range(tf.shape(inputs)[-1])
        #print("\nposition_indices: {}".format(position_indices))

        embedding_vectors = self.word_embedding_layer(inputs)
        #print("\nembedded_vectors: {}".format(embedding_vectors))

        embedding_indices = self.position_embedding_layer(position_indices)
        #print("\npositional_encoding: {}".format(embedding_indices))

        return embedding_vectors + embedding_indices


# TEST RUN OF THE CODE

# We first start with a set of English phrases that are already preprocessed and cleaned. The vectorize_layer
# creates a dictionary of words and replaces each word with its corresponding index in the dictionary. This is necessary
# because a model doesn't understand words but just vectors, numbers and matricies. We will then build the embedded
# vectors + positional vectors upon this representation.

# Let's see how it is possible to map these two sentences using this layer:
# "i am a robot"
# "you too robot"
# Steps:
# 1) We convert these two phrases to vectors of a fixed length 5 (five words in a
# sentence) through the TextVectorization layer of Keras,
# which requires a maximum vocabulary size and the required length of an output sequence for initialization.
# The output of the layer is a tensor of shape (number of sentences, output_sequence_length).
# In the following code we use adapt method to generate a vocabulary. It next creates a vectorized representation
# of the text.

#sentence_length = 5  # number of words in a sentence
#embeddingVector_length = 6  # dimension of the embedding vectors
#vocab_size = 10  # max number of different words in the vocabulary

# Sentences to be processed
#sentences = [["i am a robot"], ["you too robot"]]
#print("\nsentences: {}".format(sentences))

# We create a dataset from these sentences
#sentence_data = tf.data.Dataset.from_tensor_slices(sentences)

# Create the TextVectorization layer
#vectorize_layer = TextVectorization(output_sequence_length=sentence_length, max_tokens=vocab_size)

# Train the layer to create a vocabulary
#vectorize_layer.adapt(sentence_data)
#print("\nVocabulary: ", vectorize_layer.get_vocabulary())

# Convert all sentences to tensors
#word_tensors = convert_to_tensor(sentences, dtype=tf.string)
#print("\nword_tensors: {}".format(word_tensors))

# Use the word tensors to get vectorized phrases
#vectorized_words = vectorize_layer(word_tensors)
#print("\nVectorized words: ", vectorized_words)

# Now we compute the word embeddings and positional embeddings simultaneously using the class we have just
# defined PositionEmbeddingFixedWeights to which we pass the number of words in a sentence, how many different
# words are in the vocabulary and the desired dimension of the word embeddings + positional embeddings must be.
# We do that in order to initialize the layer. Once it is created we then pass the vectorized representation of
# the words.
#attnisallyouneed_embedding = PositionEmbeddingFixedWeights(sentence_length, vocab_size, embeddingVector_length)
#attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)
#print("\nPositional encoding + embedded words output: ", attnisallyouneed_output)
