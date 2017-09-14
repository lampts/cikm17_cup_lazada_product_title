import numpy as np
import pandas as pd
import cPickle as pickle

from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 35 
text2seq = lambda xs: map(lambda w: word2ix[w] if (w in word2ix) else word2ix['__unk__'], xs.split())
to_seq2 = lambda X: pad_sequences([text2seq(x) for x in X], maxlen=MAX_LEN)

# embedding
word2ix = pickle.load(open("../features/emb/new_word2ix.pkl", 'rb'))
embedding_matrix = np.load("../features/emb/new_laz_glove_25K.mat")

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

print '[*] Word sequences'
train_title = train_df.title_stem1u.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
test_title = test_df.title_stem1u.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
train_title = to_seq2(train_title.values)
test_title = to_seq2(test_title.values)
train_allcat = train_df.allcat_stem1.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
test_allcat = test_df.allcat_stem1.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
train_allcat = to_seq2(train_allcat.values)
test_allcat = to_seq2(test_allcat.values)
print train_title.shape, test_title.shape, train_allcat.shape, test_allcat.shape

# :: Hard coded case lookup ::
code2Idx = {'PADDING_TOKEN':0, 'numeric': 1, 'mainly_numeric':2, 'contains_digit': 3, 'size_factor':4, 'other':5}
code_embeddings = np.identity(len(code2Idx), dtype='int32')

def get_code(word, codeLookup):       
    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1
            
    digitFraction = numDigits / float(len(word))
    
    # check is this size factor
    for sz in {'ml','cm','oz','gb','inch','xl','kg','pcs','tb','inches', 'mah', 'ft', 'vac'}:
        if (sz in word) and digitFraction > 0.1:
            code = 'size_factor'
            return codeLookup[code]
    if word.isdigit(): #Is a digit
        code = 'numeric'
    elif digitFraction > 0.5:
        code = 'mainly_numeric'
    elif numDigits > 0:
        code = 'contains_digit'
    else:
        code = 'other'
    
    return codeLookup[code]

train_title_code = pad_sequences(train_df.title_stem1u.apply(lambda xs: np.asarray([get_code(w,code2Idx) for w in xs.split()])),maxlen=MAX_LEN)
test_title_code = pad_sequences(test_df.title_stem1u.apply(lambda xs: np.asarray([get_code(w,code2Idx) for w in xs.split()])),maxlen=MAX_LEN)
print train_title_code.shape, test_title_code.shape
train_allcat_code = pad_sequences(train_df.allcat_stem1.apply(lambda xs: np.asarray([get_code(w,code2Idx) for w in xs.split()])),maxlen=MAX_LEN)
test_allcat_code = pad_sequences(test_df.allcat_stem1.apply(lambda xs: np.asarray([get_code(w,code2Idx) for w in xs.split()])),maxlen=MAX_LEN)
print train_allcat_code.shape, test_allcat_code.shape

pickle.dump(train_title, open("features/train_title.p", 'wb'))
pickle.dump(train_title_code, open("features/train_title_code.p", 'wb'))
pickle.dump(test_title, open("features/test_title.p", 'wb'))
pickle.dump(test_title_code, open("features/test_title_code.p", 'wb'))
pickle.dump(train_allcat, open("features/train_allcat.p", 'wb'))
pickle.dump(train_allcat_code, open("features/train_allcat_code.p", 'wb'))
pickle.dump(test_allcat, open("features/test_allcat.p", 'wb'))
pickle.dump(test_allcat_code, open("features/test_allcat_code.p", 'wb'))

char2Idx = {}
char2Idx['_E_'] = 0
char2Idx['_U_'] = 1

for i,c in enumerate("qwertyuiopasdfghjklzxcvbnm1234567890+-*/%$.,-_:;'()"):
    char2Idx[c] = i + 2
    
CHAR_VOCAB = len(char2Idx)
CHAR_EMBED_HIDDEN_SIZE = 16
MAX_CHARLEN = 5
MAX_LEN = 35
CHAR_RNN_HIDDEN_SIZE = 16

def get_char(word,char2Idx=char2Idx,max_charlen=MAX_CHARLEN):
    cc = [char2Idx[c] if c in char2Idx else 1 for c in word]
    return pad_sequences([cc],maxlen=max_charlen)[0]

def get_char_for_sent(sent,max_len=MAX_LEN,max_charlen=MAX_CHARLEN):
    empty_word = [0]*max_charlen
    s = []
    for w in sent.split():
        s.append(get_char(w))
    if len(s) >= max_len:
        return np.asarray(s[:max_len])
    else:
        s = [empty_word]*(max_len - len(s)) + s
        return np.asarray(s)
train_title = train_df.title_stem1u.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
test_title = test_df.title_stem1u.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
train_title_char = np.asarray([get_char_for_sent(s) for s in train_title])
test_title_char = np.asarray([get_char_for_sent(s) for s in test_title])
print train_title_char.shape, test_title_char.shape

train_allcat = train_df.allcat_stem1.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
test_allcat = test_df.allcat_stem1.apply(lambda x: u' '.join(x.split()[:MAX_LEN]))
train_allcat_char = np.asarray([get_char_for_sent(s) for s in train_allcat])
test_allcat_char = np.asarray([get_char_for_sent(s) for s in test_allcat])
print train_allcat_char.shape, test_allcat_char.shape

pickle.dump(train_title_char, open("../features/train_title_char.p", 'wb'))
pickle.dump(test_title_char, open("../features/test_title_char.p", 'wb'))
pickle.dump(train_allcat_char, open("../features/train_allcat_char.p", 'wb'))
pickle.dump(test_allcat_char, open("../features/test_allcat_char.p", 'wb'))
