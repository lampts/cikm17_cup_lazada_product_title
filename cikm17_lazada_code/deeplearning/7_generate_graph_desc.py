import numpy as np
import pandas as pd
import cPickle as pickle
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

print "[*] Embedding"
word2ix = pickle.load(open("../features/emb/new_word2ix.pkl", 'rb'))
embedding_matrix = np.load("../features/emb/new_laz_glove_25K.mat")

sku2score = train_df[['sku_id', 'y_concise']].set_index('sku_id').to_dict()['y_concise']
sku_ids = train_df.sku_id.values

print "[*] Mean dist and score on train"
mean_dist = []
mean_scores = []

ds = -np.dot(X,X.T)
for i in range(0,X.shape[0]):
    dsi = ds[i]
    IX = np.argsort(dsi)
    IX = np.asarray([j for j in IX if j != i])
    IX = IX[:5]
    top5_dist = np.asarray(dsi)[IX]
    mean_dist.append(top5_dist.mean())
    top5_scores = np.asarray([sku2score[sku_ids[j]] for j in IX])
    mean_scores.append(top5_scores.mean())

mean_dist = np.asarray(mean_dist)
mean_scores = np.asarray(mean_scores)
print mean_dist.mean(), mean_scores.mean()
X_graph_train = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
print X_graph_train.shape
pickle.dump(X_graph_train, open("../features/last_week/X_graph_train.p2",'wb'))

print "[*] Mean dist and score on test"
mean_dist = []
mean_scores = []

ds = -np.dot(X_test,X.T)
for i in range(0,X_test.shape[0]):
    dsi = ds[i]
    IX = np.argsort(dsi)
    IX = np.asarray(IX)
    IX = IX[:5]
    top5_dist = np.asarray(dsi)[IX]
    mean_dist.append(top5_dist.mean())
    top5_scores = np.asarray([sku2score[sku_ids[j]] for j in IX])
    mean_scores.append(top5_scores.mean())

mean_dist = np.asarray(mean_dist)
mean_scores = np.asarray(mean_scores)
print mean_dist.mean(), mean_scores.mean()
X_graph_test = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
print X_graph_test.shape
pickle.dump(X_graph_test, open("../features/X_graph_test.p2",'wb'))

# compute tfidf vectors with scikits
v = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), max_features = 5000, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=1.0, min_df=2)

v.fit(train_df.title_stem2.values)
print "[*] Mean dist and score on train"
mean_dist = []
mean_scores = []

X = v.transform(train_df.title_stem2.values)
X_test = v.transform(test_df.title_stem2.values)

ds = -np.dot(X,X.T)
for i in range(0,X.shape[0]):
    dsi = ds[i].todense().tolist()[0]
    IX = np.argsort(dsi)
    IX = np.asarray([j for j in IX if j != i])
    IX = IX[:10]
    top5_dist = np.asarray(dsi)[IX]
    mean_dist.append(top5_dist.mean())
    top5_scores = np.asarray([sku2score[sku_ids[j]] for j in IX])
    mean_scores.append(top5_scores.mean())

mean_dist = np.asarray(mean_dist)
mean_scores = np.asarray(mean_scores)
print mean_dist.mean(), mean_scores.mean()
X_graph_train = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
pickle.dump(X_graph_train, open("../features/X_graph_train.p10",'wb'))

print "[*] Mean dist and score on test"
mean_dist = []
mean_scores = []

ds = -np.dot(X_test,X.T)
for i in range(0,X_test.shape[0]):
    dsi = ds[i].todense().tolist()[0]
    IX = np.argsort(dsi)
    IX = IX[:10]
    top5_dist = np.asarray(dsi)[IX]
    mean_dist.append(top5_dist.mean())
    top5_scores = np.asarray([sku2score[sku_ids[j]] for j in IX])
    mean_scores.append(top5_scores.mean())

mean_dist = np.asarray(mean_dist)
mean_scores = np.asarray(mean_scores)
print mean_dist.mean(), mean_scores.mean()
X_graph_test = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
print X_graph_test.shape
pickle.dump(X_graph_test, open("../features/X_graph_test.p10",'wb'))

# compute tfidf vectors with scikits
v = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), max_features = 5000, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=1.0, min_df=2)

train_df.fillna('',inplace=True)
v.fit(train_df.desc_stem1u.values)
test_df.fillna('',inplace=True)
X = v.transform(train_df.desc_stem1u.values)
X_test = v.transform(test_df.desc_stem1u.values)
tfsum = X.sum(axis=1)
tfmean = X.mean(axis=1)
print tfsum.shape, tfmean.shape

tfsum_test = X_test.sum(axis=1)
tfmean_test = X_test.mean(axis=1)
print tfsum_test.shape, tfmean_test.shape

X_tfdesc_train = np.hstack([tfsum,tfmean])
X_tfdesc_test = np.hstack([tfsum_test,tfmean_test])
print X_tfdesc_train.shape, X_tfdesc_test.shape

pickle.dump(X_tfdesc_train, open("../features/X_tfdesc_train.p",'wb'))
pickle.dump(X_tfdesc_test, open("../features/X_tfdesc_test.p",'wb'))

print "[*] Mean dist and score on train"
mean_dist = []
mean_scores = []

X = v.transform(train_df.desc_stem1u.values)
X_test = v.transform(test_df.desc_stem1u.values)

ds = -np.dot(X,X.T)
for i in range(0,X.shape[0]):
    dsi = ds[i].todense().tolist()[0]
    IX = np.argsort(dsi)
    IX = np.asarray([j for j in IX if j != i])
    IX = IX[:10]
    top5_dist = np.asarray(dsi)[IX]
    mean_dist.append(top5_dist.mean())
    top5_scores = np.asarray([sku2score[sku_ids[j]] for j in IX])
    mean_scores.append(top5_scores.mean())

mean_dist = np.asarray(mean_dist)
mean_scores = np.asarray(mean_scores)
print mean_dist.mean(), mean_scores.mean()

X_gdesc_train = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
print X_gdesc_train.shape
pickle.dump(X_gdesc_train, open("../features/X_gdesc_train.p20",'wb'))

print "[*] Mean dist and score on test"
mean_dist = []
mean_scores = []

ds = -np.dot(X_test,X.T)
for i in range(0,X_test.shape[0]):
    dsi = ds[i].todense().tolist()[0]
    IX = np.argsort(dsi)
    IX = IX[:10]
    top5_dist = np.asarray(dsi)[IX]
    mean_dist.append(top5_dist.mean())
    top5_scores = np.asarray([sku2score[sku_ids[j]] for j in IX])
    mean_scores.append(top5_scores.mean())

mean_dist = np.asarray(mean_dist)
mean_scores = np.asarray(mean_scores)
print mean_dist.mean(), mean_scores.mean()

X_gdesc_test = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
print X_gdesc_test.shape
pickle.dump(X_gdesc_test, open("../features/X_gdesc_test.p20",'wb'))


