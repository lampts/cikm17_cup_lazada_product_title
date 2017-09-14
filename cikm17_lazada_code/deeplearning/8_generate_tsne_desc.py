from __future__ import division
__author__ = 'lamp'
__version__ = "0.2"

# this is desinged to extract dense features.

import numpy as np
import pandas as pd
import cPickle as pickle

from nltk.stem.porter import *
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import wordnet
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import fuzz

import spacy
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tsne import bh_sne
from sklearn.metrics.pairwise import cosine_similarity

train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

train_df.fillna('',inplace=True)
test_df.fillna('',inplace=True)

print "[*] TSNE features"
tfidf_vectorizer = TfidfVectorizer(input='content', 
        encoding='utf-8', decode_error='replace', strip_accents='unicode', 
        lowercase=True, analyzer='word', stop_words='english', 
        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
        ngram_range=(1, 2), max_features = 5000, 
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
        max_df=0.8, min_df=2).\
        fit(train_df.desc_stem1.values.tolist()+ test_df.desc_stem1.values.tolist())
    
X_tfidf = tfidf_vectorizer.transform(train_df.desc_stem1.values.tolist() + test_df.desc_stem1.values.tolist())
svd = TruncatedSVD(n_components=200)
X_svd = svd.fit_transform(X_tfidf)
X_scaled = StandardScaler().fit_transform(X_svd)
X_tsne = bh_sne(X_scaled)
X_train_title_tsne = X_tsne[:len(train_df)]
X_test_title_tsne = X_tsne[len(train_df):]

print "TSNE shape: ", X_train_title_tsne.shape, X_test_title_tsne.shape
pickle.dump(X_train_title_tsne,open("../features/X_desc_tsne_train.p",'wb'))
pickle.dump(X_test_title_tsne,open("../features/X_desc_tsne_test.p",'wb'))

