import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from keras.preprocessing.sequence import pad_sequences
import cPickle as pickle

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

char2Idx2 = {}
char2Idx2['_E_'] = 0
char2Idx2['_U_'] = 1

for i,c in enumerate("qwertyuiopasdfghjklzxcvbnm1234567890"):
    char2Idx2[c] = i + 2
    
print len(char2Idx2)

sku_train = []
for i,r in train_df.iterrows():
    sku = r['sku_id'].lower()
    s = [char2Idx2[c] if c in char2Idx2 else 1 for c in sku]
    sku_train.append(s)
    
sku_test = []
for i,r in test_df.iterrows():
    sku = r['sku_id'].lower()
    s = [char2Idx2[c] if c in char2Idx2 else 1 for c in sku]
    sku_test.append(s)

print len(sku_train),len(sku_test)
sku_train = pad_sequences(sku_train,maxlen=20)
sku_test = pad_sequences(sku_test,maxlen=20)

pickle.dump(sku_train, open("../features/sku_train.p", 'wb'))
pickle.dump(sku_test, open("../features/sku_test.p", 'wb'))
