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

from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sess = tf.InteractiveSession()
K.set_session(sess)


X_train = pickle.load(open("../features/X_train_39.p", 'rb'))
X_test = pickle.load(open("../features/X_test_39.p",'rb'))
print X_train.shape, X_test.shape

word2ix = pickle.load(open("../features/emb/word2ix_33K.pkl", 'rb'))
embedding_matrix = np.load("../features/emb/new_laz_glove_33K.mat")

print len(word2ix), embedding_matrix.shape

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
train_title = pickle.load(open("../features/train_title.p",'rb'))
train_title_code = pickle.load(open("../features/train_title_code.p",'rb'))
train_title_char = pickle.load(open("../features/train_title_char.p",'rb'))
test_title = pickle.load(open("../features/test_title.p2",'rb'))
test_title_code = pickle.load(open("../features/test_title_code.p2",'rb'))
test_title_char = pickle.load(open("../features/test_title_char.p2",'rb'))
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

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

y_clarity = train_df.y_clarity.values
y_clarity_log = np.log1p(y_clarity)
test_sku  = test_df.sku_id.values

feature_select = []
for i in range(X_train.shape[1]):
    score = spearmanr(train_df.y_clarity,X_train[:,i])[0]
    if np.abs(score) > 0.02:
        feature_select.append(i)
print len(feature_select)

print "[*] Training on selected and sequences"
VERSION = 823
X_select_train = X_train[:,feature_select]
X_select_test = X_test[:,feature_select]
print X_select_train.shape, X_select_test.shape

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=75)
    
print "[*] Load new feature(7)"
X_newfeature_tr = pickle.load(open("../features/X_newfeature_tr.p", 'rb'))
X_newfeature_te = pickle.load(open("../features/X_newfeature_te.p2",'rb'))
print X_newfeature_tr.shape, X_newfeature_te.shape

def rectify_unit2(text):
     # unit
    text = re.sub(r"(\d+)kgs ", lambda m: '100' + ' kg ', text)
    text = re.sub(r"(\d+)kg ", lambda m: '100' + ' kg ', text)
    text = re.sub(r"(\d+)cm ", lambda m: '100' + ' cm ', text)
    text = re.sub(r"(\d+)mm ", lambda m: '100' + ' mm ', text)
    text = re.sub(r"(\d+)ml ", lambda m: '100' + ' ml ', text)
    text = re.sub(r"(\d+)mg ", lambda m: '100' + ' mg ', text)
    text = re.sub(r"(\d+)pcs ", lambda m: '100' + ' pc ', text)
    text = re.sub(r"(\d+)pc ", lambda m: '100' + ' pc ', text)
    text = re.sub(r"(\d+)gb ", lambda m: '100' + ' gb ', text)
    text = re.sub(r"(\d+)tb ", lambda m: '100' + ' tb ', text)
    text = re.sub(r"(\d+)mb ", lambda m: '100' + ' mb ', text)
    text = re.sub(r"(\d+)mah ", lambda m: '100' + ' mah ', text)
    text = re.sub(r"(\d+)mhz ", lambda m: '100' + ' mhz ', text)
    text = re.sub(r"(\d+)vac ", lambda m: '100' + ' vac ', text)
    text = re.sub(r"(\d+)inch ", lambda m: '100' + ' inch ', text)
    text = re.sub(r"(\d+)inches ", lambda m: '100' + ' inch ', text)
    text = re.sub(r"(\d+)ft ", lambda m: '100' + ' ft ', text)
    return text

train_df.title_stem1u = train_df.title_stem1.apply(rectify_unit2)
test_df.title_stem1u = test_df.title_stem1.apply(rectify_unit2)

vectorizer = TfidfVectorizer(ngram_range=(1,1),min_df=2,max_df=0.8,max_features=1000,lowercase=True,analyzer='word')
vectorizer.fit(train_df.title_stem1u.values.tolist() + test_df.title_stem1u.values.tolist() + \
               train_df.allcat_stem1.values.tolist() + test_df.allcat_stem1.values.tolist()
              )
Xtitle_train = vectorizer.transform(train_df.title_stem1u.values.tolist()).toarray()
Xtitle_test = vectorizer.transform(test_df.title_stem1u.values.tolist()).toarray()
Xallcat_train = vectorizer.transform(train_df.allcat_stem1.values.tolist()).toarray()
Xallcat_test = vectorizer.transform(test_df.allcat_stem1.values.tolist()).toarray()
print Xtitle_train.shape, Xtitle_test.shape
print Xallcat_train.shape, Xallcat_test.shape

Xcos_train = []
for i in range(Xtitle_train.shape[0]):
    Xcos_train.append(cosine_similarity(Xtitle_train[i].reshape(1,-1),Xallcat_train[i].reshape(1,-1))[0])
    
Xcos_train = np.asarray(Xcos_train)
print Xcos_train.shape

Xcos_test = []
for i in range(Xtitle_test.shape[0]):
    Xcos_test.append(cosine_similarity(Xtitle_test[i].reshape(1,-1),Xallcat_test[i].reshape(1,-1))[0])
    
Xcos_test = np.asarray(Xcos_test)
print Xcos_test.shape

X_tf_train = np.hstack([Xtitle_train,Xcos_train,Xallcat_train])
X_tf_test = np.hstack([Xtitle_test,Xcos_test,Xallcat_test])

print X_tf_train.shape, X_tf_test.shape

import xgboost as xgb

params = {
        'seed': 75,
        'colsample_bytree': 0.85,
        'silent': 1,
        'subsample': 0.85,
        'learning_rate': 0.03,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 5,
        'booster': 'gbtree',
        'base_score': y_clarity.mean(),
        'min_child_weight': 5,
        'scale_pos_weight': 1
        }

print "[*] XGB"
VERSION = 823
LAYER_NB = 19
print X_tf_train.shape, X_tf_test.shape

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=75)
model_count = 0
xg_preds = []
xg_cv_scores = []

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
    best_model_path = "../models/clarity_{}_{}.h5".format(VERSION,model_count)
    model.load_weights(best_model_path)
    feature_extractor = Model(inputs=model.input, outputs=model.layers[LAYER_NB].output)
    
    print "[*] Extract NN features"
    train_features = feature_extractor.predict([title_tr,title_code_tr,title_char_tr,cat_tr,cat_code_tr,cat_char_tr,f_tr], batch_size=350)
    valid_features = feature_extractor.predict([title_val,title_code_val,title_char_val,cat_val,cat_code_val,cat_char_val,f_val], batch_size=350)
    test_features = feature_extractor.predict([test_title,test_title_code,test_title_char,test_allcat,test_allcat_code,test_allcat_char,X_select_test], batch_size=350)
    print train_features.shape, valid_features.shape, test_features.shape
    
    print "[*] TFIDF features"
    X_tf_tr,X_tf_val = X_tf_train[idx_train],X_tf_train[idx_val]
    X_new_tr,X_new_val = X_newfeature_tr[idx_train],X_newfeature_tr[idx_val]
    
    d_train = xgb.DMatrix(np.hstack([X_new_tr,X_tf_tr,train_features]), label=y_tr)
    d_valid = xgb.DMatrix(np.hstack([X_new_val,X_tf_val,valid_features]), label=y_val)
    d_test = xgb.DMatrix(np.hstack([X_newfeature_te,X_tf_test,test_features]))
    
    watchlist = [(d_train, 'train'), (d_valid, 'eval')]
    clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, verbose_eval=50)
    preds_val = clf.predict(d_valid, ntree_limit=clf.best_ntree_limit)
    xg_cv_scores.append(metrics.log_loss(y_val, preds_val))
    print xg_cv_scores
    pred = clf.predict(d_test, ntree_limit=clf.best_ntree_limit)
    xg_preds.append(pred.ravel())
    model_count += 1
    print xg_cv_scores
    print '-'*79
    
print xg_cv_scores
print np.mean(xg_cv_scores)

xg_preds = np.asarray(xg_preds)
xg_preds = xg_preds.T
y_clarity_test2 = xg_preds.mean(axis=1)
print y_clarity_test2.shape, y_clarity_test2.mean()

# fill all test, filter valid =1, merge with valid df, save to file
out_df['y_clarity_en'] = y_clarity_test2
out_df1 = out_df[out_df.is_valid==1]
out_df1 = out_df1[['sku_id','y_clarity_en']]
submit = pd.read_csv("../data/data_valid.csv",encoding='utf-8', names=['country', 'sku_id', 'title', 'cat1', 'cat2', 'cat3', 'description', 'price', 'product_type'])
submit = submit[['sku_id']]
submit = pd.merge(submit,out_df1,on='sku_id')
np.savetxt('../submit/clarity_valid.predict', submit.y_clarity_en.values, fmt='%.5f')
