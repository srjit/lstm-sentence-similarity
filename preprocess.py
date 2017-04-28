import nltk
## from unicodedata import category

stopwords = nltk.corpus.stopwords.words('english')
word_lemmatizer = nltk.WordNetLemmatizer()


__author__ = "Sreejith Sreekumar"
__version__ = "1.1"



def lemmatize(word):
    """
    
    Arguments:
    - `tokens`:
    """
    return word_lemmatizer.lemmatize(word)

    


def preprocess(line):
    """
    removes stopwords from the array of tokens received
    
    Arguments:
    - `tokens`:
    """
    tokens = line.lower().split()
    processed_words = [lemmatize(word) for word in tokens if not word in stopwords and len(word)>3]
    return " ".join(processed_words)

    
    
def process(line):
    """
    
     Only preprocessing currently - Stop word removal
     Not adding bigrams and stuff
    
    Arguments:
    - `line`:
    """
    ## tokens = tokenize(line)
    return preprocess(str(line))
