from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Dropout
from keras.engine.topology import Merge
from keras.layers.core import Reshape
from keras.layers.advanced_activations import PReLU

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

## See the layers
## https://groups.google.com/forum/#!topic/keras-users/tIdI-j8XjM0

import reader

__author__ = "Sreejith Sreekumar"
__version__ = "1.1"


[qn1, qn2] = reader.get_formatted_data()
responses = reader.get_response()

input_layer_matrix = reader.get_embedding_matrix_input()
word_indices = reader.get_word_index()


## Keras Sequential Model
model_qn1 = Sequential()
model_qn1.add(Embedding(len(word_indices) + 1,
                     300,
                     weights=[input_layer_matrix],
                     input_length=40,
                     trainable=False))
model_qn1.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))


model_qn2 = Sequential()
model_qn2.add(Embedding(len(word_indices) + 1,
                     300,
                     weights=[input_layer_matrix],
                     input_length=40,
                     trainable=False))
model_qn2.add(LSTM(300, dropout_W=0.2, dropout_U=0.2))


mixed_model = Sequential()
mixed_model.add(Merge([model_qn1, model_qn2], mode='concat'))
print(mixed_model.layers[-1].output_shape)

mixed_model.add(BatchNormalization())
print(mixed_model.layers[-1].output_shape)

# 
# mixed_model.add(Reshape((1, 600)))
# print(mixed_model.layers[-1].output_shape)
# mixed_model.add(Reshape((1, 600)))
# print(mixed_model.layers[-1].output_shape)pp

#mixed_model.add(LSTM(600, dropout_W=0.2, dropout_U=0.2))


mixed_model.add(Dense(300))
mixed_model.add(PReLU())
mixed_model.add(Dropout(0.2))
mixed_model.add(BatchNormalization())


mixed_model.add(Dense(300))
mixed_model.add(PReLU())
mixed_model.add(Dropout(0.2))
mixed_model.add(BatchNormalization())


mixed_model.add(Dense(300))
mixed_model.add(PReLU())
mixed_model.add(Dropout(0.2))
mixed_model.add(BatchNormalization())

mixed_model.add(Dense(300))
mixed_model.add(PReLU())
mixed_model.add(Dropout(0.2))
mixed_model.add(BatchNormalization())

mixed_model.add(Dense(300))
mixed_model.add(PReLU())
mixed_model.add(Dropout(0.2))
mixed_model.add(BatchNormalization())



mixed_model.add(Dense(1))
mixed_model.add(Activation('sigmoid'))

mixed_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

history = mixed_model.fit([qn1, qn2], y=responses, batch_size=300, nb_epoch=100, verbose=1, validation_split=0.1, shuffle=True)


## Add graphs here
## http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


###############  Accuracy Calculation  ###################

## Prediction
import pandas as pd
[qn1, qn2] = reader.get_formatted_data(train=False)
responses = reader.get_response(train=False)

prediction = mixed_model.predict_classes([qn1,qn2], verbose=1)
predicted_list = pd.Series([item for sublist in prediction for item in sublist])

#actual_list = y.tolist()
frame = responses.to_frame().join(predicted_list.to_frame()).columns

# ["predicted"] = predicted_list
frame["is_correct_prediction"] = frame["duplicate"] == frame["0"]
correctly_predicted_rows = frame[frame['is_correct_prediction'] == True]
print("Accuracy : ", float(len(correctly_predicted_rows))/len(frame))
