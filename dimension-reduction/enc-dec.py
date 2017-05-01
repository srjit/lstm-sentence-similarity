
from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers.core import RepeatVector


inp_dim = 300
out_dim = 75
sequence_length = 40

model = Sequential()
model.add(GRU(inp_dim, out_dim, return_sequence=False)) # encoder
model.add(RepeatVector(sequence_length)) # Get the last output of the GRU and repeats it
model.add(GRU(out_dim, inp_dim), return_sequence=True) # decoder