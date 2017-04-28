import numpy as np



__author__ = "Sreejith Sreekumar"
__version__ = "1.1"

glove_corpus_location = '/media/sree/venus/code/glove/glove.840B.300d.txt'


def get_glove_word_vectors():
    """
    get all word vectors that glove provides
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    
    """
    glove_emb_indices = {}
    glove_file = open(glove_corpus_location)
    for line in glove_file:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            glove_emb_indices[word] = coefs
        except:
            print("Error while creating vector for word : ", word)
    glove_file.close()
    return glove_emb_indices
      
