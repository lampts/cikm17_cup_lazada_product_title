import numpy as np
import pandas as pd
import cPickle as pickle
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_json("data/train.json")
test_df = pd.read_json("data/merge/test.json")

v = TfidfVectorizer(input='content', 
        lowercase=True, analyzer='char',
        ngram_range=(2, 3), max_features = 500, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=1.0, min_df=5)
v.fit(train_df.sku_id.values)
X = v.transform(train_df.sku_id.values)
X_test = v.transform(test_df.sku_id.values)

sksum_test = X_test.sum(axis=1)
skmean_test = X_test.mean(axis=1)
X_sume_train = np.hstack([tfsum,tfmean,sksum,skmean])
X_sume_test = np.hstack([tfsum_test,tfmean_test,sksum_test,skmean_test])
print X_sume_train.shape, X_sume_test.shape

pickle.dump(X_sume_train, open("../features/X_sume_train.p",'wb'))
pickle.dump(X_sume_test, open("../features/X_sume_test.p",'wb'))

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
X_graph_test = np.hstack([mean_dist.reshape(-1,1),mean_scores.reshape(-1,1)])
print X_graph_test.shape
pickle.dump(X_graph_test, open("../features/X_graph_test.p10",'wb'))

