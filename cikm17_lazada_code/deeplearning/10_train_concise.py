import numpy as np
import pandas as pd
import cPickle as pickle

import keras
import keras.backend as K
import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from keras.layers import concatenate, multiply, add, dot, maximum
from keras.layers import merge, recurrent, Dense, Input, Dropout, TimeDistributed, Convolution1D, Lambda, Activation, RepeatVector, Flatten, Permute, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.wrappers import Bidirectional
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2, l1, l1_l2
from keras.utils import np_utils
from keras import optimizers

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rc
plt.style.use("ggplot")

from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot, plot_model
from sklearn import metrics

from keras.layers import Input,Embedding,Convolution1D, concatenate, Lambda,Conv1D,TimeDistributed

sess = tf.InteractiveSession()
K.set_session(sess)



X_train = pickle.load(open("../features/X_train_39.p", 'rb'))
X_test = pickle.load(open("../features/X_test_39.p",'rb'))
print X_train.shape, X_test.shape

word2ix = pickle.load(open("../features/emb/word2ix_33K.pkl", 'rb'))
embedding_matrix = np.load("../features/emb/new_laz_glove_33K.mat")
print len(word2ix), embedding_matrix.shape

X_graph_tr = pickle.load(open("../features/X_graph_train.p", 'rb'))
X_graph_te = pickle.load(open("../features/X_graph_test.p",'rb'))
print X_graph_tr.shape, X_graph_te.shape

X_gdesc_tr = pickle.load(open("../features/X_gdesc_train.p20", 'rb'))
X_gdesc_te = pickle.load(open("../features/X_gdesc_test.p20",'rb'))
print X_gdesc_tr.shape, X_gdesc_te.shape

X_desc_tsne_tr = pickle.load(open("../features/X_desc_tsne_train.p", 'rb'))
X_desc_tsne_te = pickle.load(open("../features/X_desc_tsne_test.p", 'rb'))
print X_desc_tsne_tr.shape,X_desc_tsne_te.shape

X_train = np.hstack([X_train,X_graph_tr,X_gdesc_tr,X_desc_tsne_tr])
X_test = np.hstack([X_test,X_graph_te,X_gdesc_te,X_desc_tsne_te])
print X_train.shape, X_test.shape

print "[*] Char and code encoding"
# :: Hard coded case lookup ::
code2Idx = {'PADDING_TOKEN':0, 'numeric': 1, 'mainly_numeric':2, 'contains_digit': 3, 'size_factor':4, 'other':5}
code_embeddings = np.identity(len(code2Idx), dtype='int32')
# char index
char2Idx = {}
char2Idx['_E_'] = 0
char2Idx['_U_'] = 1

for i,c in enumerate("qwertyuiopasdfghjklzxcvbnm1234567890+-*/%$.,-_:;'()"):
    char2Idx[c] = i + 2
    
print "[*] Sequences"
train_title = pickle.load(open("../features/train_title.p2u",'rb'))
train_title_code = pickle.load(open("../features/train_title_code.p2u",'rb'))
train_title_char = pickle.load(open("../features/train_title_char.p2u",'rb'))
test_title = pickle.load(open("../features/test_title.p2u",'rb'))
test_title_code = pickle.load(open("../features/test_title_code.p2u",'rb'))
test_title_char = pickle.load(open("../features/test_title_char.p2u",'rb'))
print train_title.shape, train_title_code.shape, train_title_char.shape
print test_title.shape, test_title_code.shape, test_title_char.shape
train_allcat = pickle.load(open("../features/train_allcat.p",'rb'))
train_allcat_code = pickle.load(open("../features/train_allcat_code.p",'rb'))
train_allcat_char = pickle.load(open("../features/train_allcat_char.p",'rb'))
test_allcat = pickle.load(open("../features/test_allcat.p2",'rb'))
test_allcat_code = pickle.load(open("../features/test_allcat_code.p2",'rb'))
test_allcat_char = pickle.load(open("../features/test_allcat_char.p2",'rb'))
print train_allcat.shape, train_allcat_code.shape, train_allcat_char.shape
print test_allcat.shape, test_allcat_code.shape, test_allcat_char.shape

def rmse_keras(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true),axis=-1))

def create_charem():
    char2Idx = {}
    char2Idx['_E_'] = 0
    char2Idx['_U_'] = 1

    for i,c in enumerate("qwertyuiopasdfghjklzxcvbnm1234567890+-*/%$.,-_:;'()"):
        char2Idx[c] = i + 2
    
    NB_FILTER_SIZE = 16 # version 1: 16
    CHAR_VOCAB = len(char2Idx)
    CHAR_EMBED_HIDDEN_SIZE = 8 # version 1: 16
    MAX_CHARLEN = 5
    MAX_LEN = 35
    CHAR_RNN_HIDDEN_SIZE = 16
    DP = 0.2
    
    input_word = Input(shape=(MAX_CHARLEN,))
    embed = Embedding(CHAR_VOCAB, CHAR_EMBED_HIDDEN_SIZE, input_length=MAX_CHARLEN, trainable=True)
    cnns = [Convolution1D(kernel_size=filt, filters=NB_FILTER_SIZE, padding='same') for filt in [2,3,5]] # version 1 2,3,5
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))

    word = embed(input_word)
    word = concatenate([maxpool(cnn(word)) for cnn in cnns])
    word_model = Model(inputs=[input_word],outputs=[word])

    char_sequences = Input(shape=(MAX_LEN, MAX_CHARLEN))
    char_outs = TimeDistributed(word_model)(char_sequences)
    charem_full = Model(inputs=[char_sequences], outputs=[char_outs])
    return charem_full

def create_model(nb_feats=25,emat=embedding_matrix):
    """Add little noise at MLP layer"""
    VOCAB = len(word2ix)
    EMBED_HIDDEN_SIZE = 300
    MAX_LEN = 35
    MAX_CHARLEN = 5
    SENT_HIDDEN_SIZE = 100
    ACTIVATION = 'elu'
    RNN_HIDDEN_SIZE = 50
    DP = 0.25
    L2 = 4e-6
    
    embed_word = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[emat], input_length=MAX_LEN, trainable=False)
    embed_code = Embedding(len(code2Idx), len(code2Idx), input_length=MAX_LEN, trainable=True)
    translate = TimeDistributed(Dense(units=SENT_HIDDEN_SIZE, activation=ACTIVATION))
    encode = Bidirectional(recurrent.LSTM(units=RNN_HIDDEN_SIZE, return_sequences=False, kernel_initializer='glorot_uniform', dropout=DP, recurrent_dropout=DP), name='my_lstm')

    # input defined: 8 tensors
    seq_title = Input(shape=(MAX_LEN,), dtype='int32') # title
    seq_title_code = Input(shape=(MAX_LEN,), dtype='int32')
    seq_title_char = Input(shape=(MAX_LEN,MAX_CHARLEN), dtype='int32')
    seq_cat= Input(shape=(MAX_LEN,), dtype='int32') # joint cats
    seq_cat_code = Input(shape=(MAX_LEN,), dtype='int32')
    seq_cat_char = Input(shape=(MAX_LEN,MAX_CHARLEN), dtype='int32')
    dense_input = Input(shape=(nb_feats,), dtype='float32')
    
    # char
    charem_full = create_charem()
    
    # rnn encode
    seq = embed_word(seq_title)
    seq = Dropout(DP)(seq)
    seq = translate(seq)
    code = embed_code(seq_title_code)
    char = charem_full(seq_title_char)
    seq = concatenate([seq,code,char])
    seq = encode(seq)
    
    seq3 = embed_word(seq_cat)
    seq3 = Dropout(DP)(seq3)
    seq3 = translate(seq3)
    code3 = embed_code(seq_cat_code)
    char3 = charem_full(seq_cat_char)
    seq3 = concatenate([seq3,code3,char3])
    seq3 = encode(seq3)
    
    # dense
    den = BatchNormalization()(dense_input)
    den = Dense(100, activation=ACTIVATION)(den)
    den = Dropout(DP)(den)

    #joint1: LOGLOSS vs RMSE
    joint = concatenate([seq,seq3,den])
    joint = Dense(units=150, activation=ACTIVATION, kernel_regularizer=l2(L2) if L2 else None, kernel_initializer='he_normal')(joint)
    joint = PReLU()(joint)
    joint = Dropout(DP)(joint)
    joint = BatchNormalization()(joint)
    
    joint = maximum([Dense(units=100, activation=ACTIVATION, kernel_regularizer=l2(L2) if L2 else None, kernel_initializer='he_normal')(joint) for _ in range(5)])
    joint = PReLU()(joint)
    joint = Dropout(DP)(joint)
    joint = BatchNormalization()(joint)

    score1 = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(L2) if L2 else None, kernel_initializer='he_normal',name='logloss')(joint)
    score2 = Dense(units=1, activation='sigmoid', kernel_regularizer=l2(L2) if L2 else None, kernel_initializer='he_normal',name='mse')(joint)

    # plug all in one
    model2 = Model(inputs=[seq_title,seq_title_code,seq_title_char,seq_cat,seq_cat_code,seq_cat_char,dense_input], outputs=[score1,score2])
    model2.compile(optimizer='nadam', loss={'logloss': 'binary_crossentropy', 'mse': 'mean_squared_error'}, \
                   loss_weights={'logloss': 0.5, 'mse': 0.5},
                   metrics=[rmse_keras])
    return model2

from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

y_clarity = train_df.y_concise.values
y_clarity_log = np.log1p(y_clarity)
test_sku  = test_df.sku_id.values

feature_select = []
for i in range(X_train.shape[1]):
    score = spearmanr(train_df.y_clarity,X_train[:,i])[0]
    if np.abs(score) > 0.02:
        feature_select.append(i)
print len(feature_select)

print "[*] Training on selected and sequences"
VERSION = 4444
X_select_train = X_train[:,feature_select]
X_select_test = X_test[:,feature_select]
print X_select_train.shape, X_select_test.shape

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=75)
model_count = 0
preds = []
preds2 = []
cv_scores = []

for idx_train, idx_val in skf.split(train_df["y_clarity"], train_df["y_clarity"]):
    print("MODEL:", model_count)
    title_tr,title_val = train_title[idx_train],train_title[idx_val]
    title_code_tr,title_code_val = train_title_code[idx_train],train_title_code[idx_val]
    title_char_tr,title_char_val = train_title_char[idx_train],train_title_char[idx_val]
    cat_tr,cat_val = train_allcat[idx_train],train_allcat[idx_val]
    cat_code_tr,cat_code_val = train_allcat_code[idx_train],train_allcat_code[idx_val]
    cat_char_tr,cat_char_val = train_allcat_char[idx_train],train_allcat_char[idx_val]
    
    f_tr,f_val = X_select_train[idx_train],X_select_train[idx_val]
    y_tr,y_val = y_clarity[idx_train],y_clarity[idx_val]
    y_log_tr,y_log_val = y_clarity_log[idx_train],y_clarity_log[idx_val]
    
    model = create_model(nb_feats=X_select_train.shape[1])   
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    csv_logger = CSVLogger("../log/concise_{}_{}.log".format(VERSION,model_count))
    best_model_path = "../models/concise_{}_{}.h5".format(VERSION,model_count)
    model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, save_weights_only=True)
    hist = model.fit([title_tr,title_code_tr,title_char_tr,cat_tr,cat_code_tr,cat_char_tr,f_tr], [y_tr,y_log_tr],
                     validation_data=([title_val,title_code_val,title_char_val,cat_val,cat_code_val,cat_char_val,f_val], [y_val,y_log_val]),
                     epochs=20, batch_size=350, shuffle=True,
                     callbacks=[early_stopping, model_checkpoint,csv_logger], verbose=2)

    model.load_weights(best_model_path)
    cv_scores.append(min(hist.history["val_logloss_loss"]))
    print cv_scores
    pred,pred2 = model.predict([test_title,test_title_code,test_title_char,test_allcat,test_allcat_code,test_allcat_char,X_select_test],\
                         batch_size=350, verbose=2)
    preds.append(pred.ravel())
    preds2.append(np.expm1(pred2.ravel()))
    print "Mean: ", pred.ravel().mean()
    model_count += 1
    
print cv_scores
print np.mean(cv_scores)
print len(preds2)
preds2 = np.asarray(preds2)
preds2 = preds2.T
y_concise_test2 = preds2.mean(axis=1)
print y_concise_test2.shape, y_concise_test2.mean(), y_concise_test2.min(), y_concise_test2.max()

y_concise_test2 = np.minimum(y_concise_test2,1)
print y_concise_test2.shape, y_concise_test2.mean(), y_concise_test2.min(), y_concise_test2.max()

out_df = pd.DataFrame(dict(sku_id=test_df.sku_id.values,y_concise=y_concise_test2))
submit2 = pd.read_csv("../data/data_test.csv",encoding='utf-8', names=['country', 'sku_id', 'title', 'cat1', 'cat2', 'cat3', 'description', 'price', 'product_type'])
test_ids = set(submit2.sku_id.values.tolist())
out_df['is_valid'] = out_df.sku_id.apply(lambda x: 0 if x in test_ids else 1)

# fill all test, filter valid =1, merge with valid df, save to file
out_df1 = out_df[out_df.is_valid==0]
out_df1 = out_df1[['sku_id','y_concise']]
submit2 = submit2[['sku_id', 'title']]
submit2 = pd.merge(submit2,out_df1,on='sku_id')
np.savetxt('../submit/conciseness_test.predict', submit2.y_concise.values, fmt='%.5f')

