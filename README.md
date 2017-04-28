


Sentence similarity using Recurrent Neural Networks

Requirements :
 1) Python 3+
 2) Pandas
 3) Keras
 4) Numpy
 5) Glove glove.840B.300d.txt corpus

Instructions on running the code :

 1) In encoder.py change the "glove_corpus_location" to point to glove embeddings dataset in your filesystem.
 2) The datafile for the project is downoadable at : http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv.
 3) Run the split-creater.py script for creating the train and test datasets after pointing the source location in the script to correct folder.
 4) Place the datasets created in a folder called 'data' which is in the same level as that of the project folder. Run the functions get_data() and get_test_data() in the reader.py script
    to make sure dataframes of the train and test set are being created properly.
 5) Execute 'main.py'
 6) For model parameter tuning, change the same in main.py and re-run the script


    