from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model



import reader


[qn1, qn2] = reader.get_formatted_data()
responses = reader.get_response()

input_layer_matrix = reader.get_embedding_matrix_input()
word_indices = reader.get_word_index()



# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
# main_input = Input(shape=(100,), dtype='int32', name='main_input')

# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
# x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

x = Embedding(len(word_indices) + 1,
                     300,
                     weights=[input_layer_matrix],
                     input_length=40,
                     trainable=False)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(75)(x)

outputs = Dense(300)(lstm_out)


model = Model(input=qn1, output=outputs)
model.compile(optimizer='adam', loss='mse')
model.fit(X, X, batch_size=64, epochs=10)

