import numpy as np
import pandas as pd
from scipy.stats import spearmanr

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

def get_delta_length(df_input, column_name, cat_name="cat1", is_mean=True):
    if is_mean:
        new_feature = column_name + '_delta_mean_' + cat_name
        df_mean = df_input.groupby([cat_name])[column_name].mean()
        df_mean = df_mean.reset_index()
        df_mean.columns = [cat_name, new_feature]
        df_output = pd.merge(df_input, df_mean, on=cat_name, how="left")
    else:
        new_feature = column_name + '_delta_max_' + cat_name
        df_max = df_input.groupby([cat_name])[column_name].max()
        df_max = df_max.reset_index()
        df_max.columns = [cat_name, new_feature]
        df_output = pd.merge(df_input, df_max, on=cat_name, how="left")

    return df_output

def longest_word(s):
    s = s.split()
    if s:
        return np.max([len(w) for w in s])
    else:
        return 0
    
print "[*] Longest word"
train_df['longest_word'] = train_df.title_stem1.apply(longest_word)
test_df['longest_word'] = test_df.title_stem1.apply(longest_word)
print "Concise: ", spearmanr(train_df.y_concise,train_df.longest_word)

print "[*] fashion men"
train_df['has_fashion_men'] = train_df.title_stem1.apply(lambda x: 1 if (('fashion' in x) or ('men' in x)) else 0)
test_df['has_fashion_men'] = test_df.title_stem1.apply(lambda x: 1 if (('fashion' in x) or ('men' in x)) else 0)
print "Concise: ", spearmanr(train_df.y_concise,train_df.has_fashion_men)

print "[*] Last word = )"
train_df['last_word_bracket'] = train_df.title_stem1.apply(lambda x: 1 if x.split()[-1]==')' else 0)
test_df['last_word_bracket'] = test_df.title_stem1.apply(lambda x: 1 if x.split()[-1]==')' else 0)
print "Concise: ", spearmanr(train_df.y_concise,train_df.last_word_bracket)

import cPickle as pickle
word2ix = pickle.load(open("../features/emb/word2ix_33K.pkl", 'rb'))
embedding_matrix = np.load("../features/emb/new_laz_glove_33K.mat")

def distance_first_last_words(s):
    s = s.split()
    w1,w2 = s[0],s[-1]
    if (w1 not in word2ix) or (w2 not in word2ix):
        return 0
    else:
        return np.dot(embedding_matrix[word2ix[w1]],embedding_matrix[word2ix[w2]])
    
print "[*] Distance first last"
train_df['distance_firt_last'] = train_df.title_stem1.apply(distance_first_last_words)
test_df['distance_firt_last'] = test_df.title_stem1.apply(distance_first_last_words)
print "Concise: ", spearmanr(train_df.y_concise,train_df.distance_firt_last)

train_df['nb_spaces'] = train_df.title_stem1.str.count(' ')
test_df['nb_spaces'] = test_df.title_stem1.str.count(' ')

cat1sp = train_df.groupby(['cat1_le'])['nb_spaces'].mean().to_dict()
cat2sp = train_df.groupby(['cat2_le'])['nb_spaces'].mean().to_dict()
cat3sp = train_df.groupby(['cat3_le'])['nb_spaces'].mean().to_dict()

print "[*] Delta by cat"
train_df['delta1'] = train_df.apply(lambda r: r['nb_spaces'] - cat1sp[r['cat1_le']],axis=1)
test_df['delta1'] = test_df.apply(lambda r: r['nb_spaces'] - cat1sp[r['cat1_le']],axis=1)
train_df['delta2'] = train_df.apply(lambda r: r['nb_spaces'] - cat2sp[r['cat2_le']],axis=1)
test_df['delta2'] = test_df.apply(lambda r: r['nb_spaces'] - cat2sp[r['cat2_le']],axis=1)
train_df['delta3'] = train_df.apply(lambda r: r['nb_spaces'] - cat3sp[r['cat3_le']],axis=1)
test_df['delta3'] = test_df.apply(lambda r: r['nb_spaces'] - cat3sp[r['cat3_le']],axis=1)
print "Concise: dd1", spearmanr(train_df.y_concise,train_df.delta1)
print "Concise: dd2", spearmanr(train_df.y_concise,train_df.delta2)
print "Concise: dd3", spearmanr(train_df.y_concise,train_df.delta3)

print "Clarity: dd1", spearmanr(train_df.y_clarity,train_df.delta1)
print "Clarity: dd2", spearmanr(train_df.y_clarity,train_df.delta2)
print "Clarity: dd3", spearmanr(train_df.y_clarity,train_df.delta3)

new_features = ['delta1','delta2','delta3','distance_firt_last','last_word_bracket','has_fashion_men','longest_word']
X_newfeature_tr = train_df[new_features].as_matrix()
X_newfeature_te = test_df[new_features].as_matrix()
print X_newfeature_tr.shape, X_newfeature_te.shape

pickle.dump(X_newfeature_tr, open("../features/X_newfeature_tr.p",'wb'))
pickle.dump(X_newfeature_te, open("../features/X_newfeature_te.p2",'wb'))

    