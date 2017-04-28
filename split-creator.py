import pandas as pd
import numpy as np

data = pd.read_csv('../../../data/quora_duplicate_questions.tsv', sep='\t')



msk = np.random.rand(len(data)) < 0.9
train = data[msk]
test = data[~msk]


train.to_csv('../data/train.csv', sep='\t')
test.to_csv('../data/test.csv', sep='\t')


