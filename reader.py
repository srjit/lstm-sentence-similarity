import numpy as npp
import pandas as pd
from keras.preprocessing import sequence, text

import preprocess as pp

import encoder as enc

__author__ = "Sreejith Sreekumar"
__version__ = "1.1"


tokenizer = text.Tokenizer(nb_words=200000)

data = None

def get_formatted_data(train=True):
    """
    """

    data = None
    
    if(train) :
        data = get_data()
    else :
        data = get_test_data()
    

    data['text1_processed'] = data.text1.apply(lambda x : pp.process(x))
    data['text2_processed'] = data.text2.apply(lambda x : pp.process(x))

    ## get the tokens
    ## https://keras.io/preprocessing/text/#tokenizer
    ## https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html



    list1 = list(data.text1.values.astype(str))
    list2 = list(data.text2.values.astype(str))

    # Preprocessed text
    # Experiment - with stopwords and lemmatization - Probably not a good idea
    # list1 = list(data.text1_processed.values.astype(str))
    # list2 = list(data.text2_processed.values.astype(str))
    all_questions = list1 + list2

    tokenizer.fit_on_texts(all_questions)

    maximum_length_of_question = 40
    sequences_text1 = tokenizer.texts_to_sequences(data.text1.values)
    sequences_text1 = sequence.pad_sequences(sequences_text1, maxlen=maximum_length_of_question)

    sequences_text2 = tokenizer.texts_to_sequences(data.text2.values.astype(str))
    sequences_text2 = sequence.pad_sequences(sequences_text2, maxlen=maximum_length_of_question)

    return [sequences_text1, sequences_text2]




    
def get_embedding_matrix_input():
    """
    """
    ## Map of <word, index> of the word that
    ## keras provides
    word_indices = tokenizer.word_index

    glove_word_vectors = enc.get_glove_word_vectors()

    ## Glove Embedding for every word from the word_indices
    input_layer_matrix = np.zeros((len(word_indices) + 1, 300))
    for word, index in word_indices.items():
        word_vector = glove_word_vectors.get(word)
        if word_vector is not None:
            input_layer_matrix[index] = word_vector

    return input_layer_matrix
    

def get_response(train=True):
    data = None
    if(train):
        data = get_data()
    else :
        data = get_test_data()
    response = data.duplicate
    #response_encoded = np_utils.to_categorical(response)
    return response
    

def get_data():
    global data
    if data is None :
        data = pd.read_csv('../data/train.csv', sep='\t')

    return data



def get_test_data():
    global data
    if data is None :
        data = pd.read_csv('../data/test.csv', sep='\t')

    return data
    

    
def get_word_index():
    """
    """
    return tokenizer.word_index