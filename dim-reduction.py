## working dimension reduction
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model

X = np.random.random((10000,300))

inputs = Input(shape=(300,))
h = Dense(75, activation='sigmoid')(inputs)


model2 = Model(input=inputs, output=h)
model2.compile(optimizer='adam',loss='mse')
out = model2.predict(X)
out.shape



## Sequential Model
## Problems - Need to get weights from output LSTM yer
## Need to reshape the input to the LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import reader

[qn1, qn2] = reader.get_formatted_data()
responses = reader.get_response()

input_layer_matrix = reader.get_embedding_matrix_input()
word_indices = reader.get_word_index()


model = Sequential()
model.add(Embedding(len(word_indices) + 1,
                     300,
                     weights=[input_layer_matrix],
                     input_length=40,
                     trainable=False))

model.add(Dense(75, input_dim=300))
model.add(LSTM(75, dropout_W=0.2, dropout_U=0.2))
model.compile(loss='mse', optimizer='adam')

model.fit(qn1, qn1, batch_size=200, nb_epoch=3, verbose=1)






## Adding Embedding layer to the working dim reduction code

import numpy as np

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.embeddings import Embedding

X = np.random.random((10000,300))

inputs = Input(shape=(300,))
h = Dense(75, activation='sigmoid')(inputs)
output = Dense(300, activation='sigmoid')(h)


model2 = Model(input=inputs, output=output)
model2.compile(optimizer='adam',loss='mse')
out = model2.predict(X)
out.shape







## http://danielhnyk.cz/predicting-sequences-vectors-keras-using-rnn-lstm/

from keras.models import Sequential  
from keras.layers.core import Dense, Activation  
from keras.layers.recurrent import LSTM

in_out_neurons = 2  
hidden_neurons = 300

model = Sequential()  
model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))  
model.add(Dense(hidden_neurons, in_out_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")





## scrap - staging code

## working dimension reduction
import numpy as np

from keras.layers.recurrent import LSTM
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Reshape

X = np.random.random((10000,300))

inputs = Input(shape=(300,))
flatten = Reshape((1,300)) (inputs)
h = LSTM(75, activation='sigmoid')(flatten)
model2 = Model(input=inputs, output=h)
model2.compile(optimizer='adam',loss='mse')
out = model2.predict(X)
out.shape
