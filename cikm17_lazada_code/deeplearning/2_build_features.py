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

print "[*] Loading packages"
nlp = spacy.load('en')
ratio_features = ['nb_words','nb_chars','nb_oov_stop', 'mean_char', 'ratio_noun', 'ratio_adj', 'ratio_oov', 'ratio_nonascii', 'ratio_num']
predictive_features = ['location_sim_max', 'brand_sim_max', 'unit_sim_max', 'ship_sim_max', 'color_sim_max', 'newsexy_sim_max', 'cat_sim_max', 'cat_sim_mean', 'nb_measure_unit', 'sku_length', 'has_new', 'has_sexy', 'title_entropy_char', 'title_entropy_word', 'nb_dup_title', 'cat2_le']

toker = TreebankWordTokenizer()
lemmer = wordnet.WordNetLemmatizer()

def text_preprocessor(x,keep_origin=False):
    '''
    Get one string and clean\lemm it
    '''
    tmp = unicode(x)
    tmp = BeautifulSoup(tmp)
    tmp = tmp.getText(' ')
    tmp = tmp.lower()
    if keep_origin:
        x_cleaned = tmp
    else:
        x_cleaned = text_replace_num(text_replace_size(text_punc_seperation(tmp)))
    tokens = toker.tokenize(x_cleaned)
    return u" ".join([lemmer.lemmatize(z) for z in tokens])


num_re = re.compile(ur'[\.\+\-\$\u20ac]?\d+((\.\d+)|(,\d+))*(st|nd|rd|th)?', re.I | re.UNICODE)
size_re = re.compile(ur'[\.\+\-\$\u20ac]?\d+((\.\d+)|(,\d+)|(x\d+))*(ml|cm|oz|gb|inch|inches|xl|kg|mg|pcs|pc|tb|m|ft|mah|a|vac)', re.I | re.UNICODE)

def camel_split(s):
    """ 
    Is it ironic that this function is written in camel case, yet it
    converts to snake case? hmm..
    """
    _underscorer1 = re.compile(r'(.)([A-Z][a-z]+)')
    _underscorer2 = re.compile('([a-z0-9])([A-Z])')
    subbed = _underscorer1.sub(r'\1 \2', s)
    return _underscorer2.sub(r'\1 \2', subbed)

def size_split(s):
    _underscorer1 = re.compile(r'(.)([0-9](ml|cm|oz|gb|inch|inches|xl|kg|mg|pcs|pc|tb|m|mah)[a-z]+)')
    _underscorer2 = re.compile(r'([a-z0-9])([A-Z])')
    subbed = _underscorer1.sub(r'\1 \2', s)
    return _underscorer2.sub(r'\1 \2', subbed)

def text_replace_size(s):
    """
    100ml -> 100 ml
    """
    s = unicode(s)
    s = u' '.join([u' '.join(re.findall(r'[A-Za-z]+|\d+', w)) if size_re.match(w) else w for w in s.strip().split()])
    return s

def text_replace_num(s):
    """
    123 -> 0
    """
    s = unicode(s)
    s = re.sub(r'(\w+)-(\w+)', r'\1 - \2', s) # fixed - by digit and named
    s = u' '.join(["0" if num_re.match(w) else w for w in s.strip().split()])
    return s

def text_punc_seperation(s):
    """
    #this function will convert text to lowercase and will disconnect punctuation and special symbols from words
      function normalize_text {
        awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
        -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
      }
    """
    s = unicode(s)
    s = re.sub(r"\.\s", " ", s, re.I | re.UNICODE)
    s = re.sub(r"\.@", "@", s, re.I | re.UNICODE)
    s = re.sub(r"\'", " ", s, re.I | re.UNICODE)
    s = re.sub(r"\"", ' " ', s, re.I | re.UNICODE)
    s = re.sub(r",", ' ', s, re.I | re.UNICODE)
    s = re.sub(r"\(", " ( ", s, re.I | re.UNICODE)
    s = re.sub(r"\)", " ) ", s, re.I | re.UNICODE)
    s = re.sub(r"\[", " [ ", s, re.I | re.UNICODE)
    s = re.sub(r"\]", " ] ", s, re.I | re.UNICODE)
    s = re.sub(r"!", " ! ", s, re.I | re.UNICODE)
    s = re.sub(r"\?", " ? ", s, re.I | re.UNICODE)
    s = re.sub(r";", " ", s, re.I | re.UNICODE)
    s = re.sub(r":\s", " : ", s, re.I | re.UNICODE)
    s = re.sub(ur"\u201C", " ", s, re.I | re.UNICODE)
    s = re.sub(ur"\u201D", " ", s, re.I | re.UNICODE)
    s = re.sub(r"/", " ", s, re.I | re.UNICODE)
    s = re.sub(ur"`", " ", s, re.I | re.UNICODE)
    s = re.sub(r"-", " ", s, re.I| re.UNICODE)
    s = re.sub(r"#", " ", s, re.I | re.UNICODE)
    s = re.sub(r">", " > ", s, re.I | re.UNICODE)
    # s = re.sub(r"@", "@", s, re.I| re.UNICODE)
    # s = re.sub(r"_", " ", s, re.I| re.UNICODE)
    s = re.sub(r"\|", " | ", s, re.I | re.UNICODE)
    s = re.sub(r"\|", " | ", s, re.I | re.UNICODE)
    s = re.sub(r"=", " = ", s, re.I | re.UNICODE)
    s = re.sub(r"\+", " + ", s, re.I | re.UNICODE)
    s = re.sub(r"\*", " * ", s, re.I | re.UNICODE)
    s = re.sub(r"%", " % ", s, re.I | re.UNICODE)
    s = re.sub(ur"\u2019", " ", s, re.I | re.UNICODE)
    s = re.sub(ur"\u2018", " ", s, re.I | re.UNICODE)
    s = re.sub(ur"\u2026", " ", s, re.I | re.UNICODE)
    s = re.sub(ur"\u26f9", " ", s, re.I | re.UNICODE)
    s = re.sub(r"\n", " ", s, re.I | re.UNICODE)
    s = re.sub(r"\r", " ", s, re.I | re.UNICODE)

    # remove some exuberancy " " character

    s = ' '.join([w.replace('@','') if w.startswith('@') or w.startswith('__') else w.replace('-', '').replace('_', '') for w in s.strip().split()])
    return s

preprocessing = lambda x: text_punc_seperation(text_replace_num(x))

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

def get_dense_features(sent):
    """ratio: noun/nb_words, oov+stop, adj,num, avg char per word"""
    SAFEDIV=2e-6
    doc = nlp(sent.lower())
    nb_noun = 0
    nb_adj = 0
    nb_oov = 0
    nb_stop = 0
    nb_bracket = 0
    nb_nonalpha = 0
    nb_num = 0
    nb_upper = 0
    nb_nonascii = 0
    for i in doc:
        if i.pos_ == 'NOUN': nb_noun += 1
        if i.pos_ == 'ADJ': nb_adj += 1
        if i.is_oov: nb_oov += 1
        if i.is_stop: nb_stop += 1
        if i.is_bracket: nb_bracket += 1
        if not i.is_alpha: nb_nonalpha += 1
        if i.is_digit or i.like_num or i.pos_ == 'NUM': nb_num += 1
        if i.orth_.isupper(): nb_upper += 1
        if not i.is_ascii: nb_nonascii += 1
    nb_words = len(sent.split())
    nb_chars = len(sent.replace(' ',''))
    return (nb_words,nb_chars,nb_oov+nb_stop,nb_chars /(nb_words+SAFEDIV),nb_noun / (nb_words+SAFEDIV),nb_adj/(nb_words+SAFEDIV) ,(nb_oov+nb_stop)/(nb_words+SAFEDIV),(nb_bracket + nb_nonalpha + nb_nonascii)/(nb_words+SAFEDIV),nb_num/(nb_words+SAFEDIV))

def H(terms,count_dict,n):
    s = 0.0
    for word in terms:
        if word in count_dict:
            s+=-(1.*count_dict[word]/n)*np.log(1.*count_dict[word]/n)
    return s

measure_re = re.compile(ur'[\.\+\-\$\u20ac]?\d+((\.\d+)|(,\d+)|(x\d+))*(ml|cm|oz|gb|inch|inches|xl|kg|mg|pcs|pc|tb|m|ft|mah|a|vac)', re.I | re.UNICODE)
def count_nb_like_measure(s):
    s = s.lower()
    out = 0
    for w in s.split():
        if measure_re.match(w):
            out += 1
    return out

def hierarchy_similarity(s1,s2,mode=1):
    """mode 1-max, 2-mean, return similarity score based on words"""
    s1,s2 = s1.lower(),s2.lower()
    scores = []
    for w1 in s1.split():
        for w2 in s2.split():
            if (w1 in word2ix) and (w2 in word2ix):
                scores.append(np.dot(embedding_matrix[word2ix[w1]],embedding_matrix[word2ix[w2]]))
            else:
                scores.append(0)
    if not scores:
        return 0
    if mode ==1:
        return np.max(scores)
    else:
        return np.mean(scores)

def get_sentence_vector(sent):
    v = np.zeros((300,))
    c = 0
    for w in sent.strip().split():
        if w in word2ix:
            v = v + embedding_matrix[word2ix[w]]
            c += 1
    v = v / (c + 0.0001)
    return v

stop_words = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
        "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be",
        "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "co",
        "con", "could", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight",
        "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill", "find", "fire", "first", "five", "for",
        "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
        "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself",
        "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest",
        "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made",
        "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much",
        "must", "my", "myself", "name", "namely", "neither", "nevertheless", "next", "nine", "nobody", "now", "nowhere",
        "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
        "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "rather", "re", "same", "see",
        "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some",
        "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take",
        "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though",
        "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve",
        "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
        "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon",
        "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will",
        "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
        ]

STOP_WORDS = set(stop_words)
SAFE_DIV = 0.0001

def get_token_features(q1, q2):
    """q1,q2: stem, lowercase"""
    token_features = [0.0]*5

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[2] = fuzz.token_set_ratio(q1, q2)
    token_features[3] = fuzz.token_sort_ratio(q1, q2)
    token_features[4] = fuzz.QRatio(q1, q2)
    return token_features

def rectify_unit(text):
     # unit
    text = re.sub(r"(\d+)kgs ", lambda m: m.group(1) + ' kg ', text)
    text = re.sub(r"(\d+)kg ", lambda m: m.group(1) + ' kg ', text)
    text = re.sub(r"(\d+)cm ", lambda m: m.group(1) + ' cm ', text)
    text = re.sub(r"(\d+)mm ", lambda m: m.group(1) + ' mm ', text)
    text = re.sub(r"(\d+)ml ", lambda m: m.group(1) + ' ml ', text)
    text = re.sub(r"(\d+)mg ", lambda m: m.group(1) + ' mg ', text)
    text = re.sub(r"(\d+)pcs ", lambda m: m.group(1) + ' pc ', text)
    text = re.sub(r"(\d+)pc ", lambda m: m.group(1) + ' pc ', text)
    text = re.sub(r"(\d+)gb ", lambda m: m.group(1) + ' gb ', text)
    text = re.sub(r"(\d+)tb ", lambda m: m.group(1) + ' tb ', text)
    text = re.sub(r"(\d+)mb ", lambda m: m.group(1) + ' mb ', text)
    text = re.sub(r"(\d+)mah ", lambda m: m.group(1) + ' mah ', text)
    text = re.sub(r"(\d+)mhz ", lambda m: m.group(1) + ' mhz ', text)
    text = re.sub(r"(\d+)vac ", lambda m: m.group(1) + ' vac ', text)
    text = re.sub(r"(\d+)inch ", lambda m: m.group(1) + ' inch ', text)
    text = re.sub(r"(\d+)inches ", lambda m: m.group(1) + ' inch ', text)
    text = re.sub(r"(\d+)ft ", lambda m: m.group(1) + ' ft ', text)
    return text

print "[*] Embedding"
word2ix = pickle.load(open("../features/emb/new_word2ix.pkl", 'rb'))
embedding_matrix = np.load("../features/emb/new_laz_glove_25K.mat")

print "[*] Dataframe"
train_df = pd.read_json("../data/train.json")
test_df = pd.read_json("../data/merge/test.json")

y_clarity = train_df.y_clarity.values
y_clarity_log = np.log1p(y_clarity)
y_concise = train_df.y_concise.values
y_concise_log = np.log1p(y_concise)
test_sku  = test_df.sku_id.values

import nltk
nltk.download('wordnet')

print "[*] Df transform"
train_df['title_stem1'] = train_df.title.apply(lambda x:text_preprocessor(x,keep_origin=True))
train_df['title_stem2'] = train_df.title.apply(lambda x:text_preprocessor(x,keep_origin=False))
test_df['title_stem1'] = test_df.title.apply(lambda x:text_preprocessor(x,keep_origin=True))
test_df['title_stem2'] = test_df.title.apply(lambda x:text_preprocessor(x,keep_origin=False))
train_df['allcat'] = train_df.apply(lambda r: u" ".join([r['cat1'],r['cat2'],r['cat3']]), axis=1)
test_df['allcat'] = test_df.apply(lambda r: u" ".join([r['cat1'],r['cat2'],r['cat3']]), axis=1)
train_df['allcat_stem1'] = train_df.allcat.apply(lambda x: text_preprocessor(x,keep_origin=True))
test_df['allcat_stem1'] = test_df.allcat.apply(lambda x: text_preprocessor(x,keep_origin=True))
train_df['desc_stem1'] = train_df.description.apply(lambda x:text_preprocessor(x,keep_origin=True))
test_df['desc_stem1'] = test_df.description.apply(lambda x:text_preprocessor(x,keep_origin=True))

print "[*] Label encoding"
cat_cols = ['cat1', 'cat2', 'cat3', 'country', 'product_type']

for c in cat_cols:
    lbe = LabelEncoder()
    lbe.fit(train_df[c].values.tolist() + test_df[c].values.tolist())
    train_df[c+'_le'] = lbe.transform(train_df[c].values.tolist())
    test_df[c+'_le'] = lbe.transform(test_df[c].values.tolist())
    
print "[*] Duplicate ratio"
counter = CountVectorizer(ngram_range=(1,2),max_features=10000,stop_words=None,lowercase=True)
counter.fit(train_df.title_stem1.values.tolist()+test_df.title_stem1.values.tolist())
train_df['nb_dup_title'] = train_df.title_stem1.apply(lambda x: np.sum(counter.transform([x]).toarray()>1))
test_df['nb_dup_title'] = test_df.title_stem1.apply(lambda x: np.sum(counter.transform([x]).toarray()>1))

print "[*] Entroy by char/word"
counter = CountVectorizer(ngram_range=(1,1), analyzer='word')
X = counter.fit_transform(train_df.title_stem1.values.tolist() + test_df.title_stem1.values.tolist())
char_freq_df = pd.DataFrame({'word': counter.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
count_dict_word = char_freq_df.set_index('word').to_dict()['occurrences']
N_words = char_freq_df.occurrences.sum()
train_df['title_entropy_word'] = train_df.title_stem1.apply(lambda x:H(x.split(),count_dict=count_dict_word,n=N_words))
test_df['title_entropy_word'] = test_df.title_stem1.apply(lambda x:H(x.split(),count_dict=count_dict_word,n=N_words))

counter = CountVectorizer(ngram_range=(1,3), analyzer='char')
X = counter.fit_transform(train_df.title_stem1.values.tolist() + test_df.title_stem1.values.tolist())
char_freq_df = pd.DataFrame({'char': counter.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
count_dict_word = char_freq_df.set_index('char').to_dict()['occurrences']
N_words = char_freq_df.occurrences.sum()
train_df['title_entropy_char'] = train_df.title_stem1.apply(lambda x:H(x.split(),count_dict=count_dict_word,n=N_words))
test_df['title_entropy_char'] = test_df.title_stem1.apply(lambda x:H(x.split(),count_dict=count_dict_word,n=N_words))

print "[*] Max pool on patterns"
train_df['location_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"china us europe",mode=1),axis=1)
test_df['location_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"china us europe",mode=1),axis=1)

# has brand sim
train_df['brand_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"sony xaomi samsung apple gucci klein crocs unilever nestle",mode=1),axis=1)
test_df['brand_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"sony xaomi samsung apple gucci klein crocs unilever nestle",mode=1),axis=1)

# has unit
train_df['unit_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"mm ml oz gb mah ft",mode=1),axis=1)
test_df['unit_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"mm ml oz gb mah ft",mode=1),axis=1)

# has ship info
train_df['ship_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"local international intl ship export",mode=1),axis=1)
test_df['ship_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"local international intl ship export",mode=1),axis=1)

# has color
train_df['color_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"gold blue dark",mode=1),axis=1)
test_df['color_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"gold blue dark",mode=1),axis=1)

# has new and sexy
train_df['newsexy_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"new sexy",mode=1),axis=1)
test_df['newsexy_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],"new sexy",mode=1),axis=1)

# relevance to cat
train_df['cat_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['allcat_stem1'],mode=1),axis=1)
train_df['cat_sim_mean'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['allcat_stem1'],mode=2),axis=1)
test_df['cat_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['allcat_stem1'],mode=1),axis=1)
test_df['cat_sim_mean'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['allcat_stem1'],mode=2),axis=1)

# count base
train_df['nb_measure_unit'] = train_df.title_stem1.apply(count_nb_like_measure)
test_df['nb_measure_unit'] = test_df.title_stem1.apply(count_nb_like_measure)

train_df['sku_length'] = train_df.sku_id.str.len()
test_df['sku_length'] = test_df.sku_id.str.len()

train_df['has_new'] = train_df.title_stem1.apply(lambda x: 1 if (('new' in x)) else 0)
test_df['has_new'] = test_df.title_stem1.apply(lambda x: 1 if (('new' in x)) else 0)

train_df['has_sexy'] = train_df.title_stem1.apply(lambda x: 1 if (('sexy' in x)) else 0)
test_df['has_sexy'] = test_df.title_stem1.apply(lambda x: 1 if (('sexy' in x)) else 0)

train_df['desc_sim_max'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['desc_stem1'],mode=1),axis=1)
train_df['desc_sim_mean'] = train_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['desc_stem1'],mode=2),axis=1)
test_df['desc_sim_max'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['desc_stem1'],mode=1),axis=1)
test_df['desc_sim_mean'] = test_df.apply(lambda r: hierarchy_similarity(r['title_stem1'],r['desc_stem1'],mode=2),axis=1)

train_df['price_cvt'] = train_df.apply(lambda r: r['price'] if r['country']=='sg' else r['price']*0.32 if r['country']=='my' else r['price']*0.027,axis=1)
test_df['price_cvt'] = test_df.apply(lambda r: r['price'] if r['country']=='sg' else r['price']*0.32 if r['country']=='my' else r['price']*0.027,axis=1)

print "[*] Split the unit measure 100cm -> 100 cm"
train_df['title_stem1u'] = train_df.title_stem1.apply(rectify_unit)
test_df['title_stem1u'] = test_df.title_stem1.apply(rectify_unit)

print "[*] Extract 9 ratio features on title"
# dense
X_ratio_train = []
for i,r in train_df.iterrows():
    vec = get_dense_features(r['title_stem1'])
    X_ratio_train.append(vec)
    
X_ratio_train = np.asarray(X_ratio_train)

X_ratio_test = []
for i,r in test_df.iterrows():
    vec = get_dense_features(r['title_stem1'])
    X_ratio_test.append(vec)
    
X_ratio_test = np.asarray(X_ratio_test)
print "Ratio shape: ", X_ratio_train.shape, X_ratio_test.shape

print "[*] TSNE features"
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words=stop_words,min_df=2,ngram_range=(1,2)).\
        fit(train_df.title_stem1.values.tolist()+ test_df.title_stem1.values.tolist())
    
X_tfidf = tfidf_vectorizer.transform(train_df.title_stem1.values.tolist() + test_df.title_stem1.values.tolist())
svd = TruncatedSVD(n_components=200)
X_svd = svd.fit_transform(X_tfidf)
X_scaled = StandardScaler().fit_transform(X_svd)
X_tsne = bh_sne(X_scaled)
X_train_title_tsne = X_tsne[:len(train_df)]
X_test_title_tsne = X_tsne[len(train_df):]

X_features_train = train_df[predictive_features].as_matrix()
X_features_test = test_df[predictive_features].as_matrix()
print "TSNE shape: ", X_features_train.shape, X_features_test.shape

X_train = np.hstack([X_ratio_train,X_features_train,X_train_title_tsne])
X_test = np.hstack([X_ratio_test,X_features_test,X_test_title_tsne])
print "Clarity shape(1): ", X_train.shape, X_test.shape

y_clarity_test = np.zeros((test_df.shape[0],))
X_train2 = np.hstack([X_train, train_df.y_clarity.reshape(-1,1)])
X_test2 = np.hstack([X_test, y_clarity_test.reshape(-1,1)])
X_train2 = np.hstack([X_train2,train_df.desc_sim_max.reshape(-1,1),train_df.desc_sim_mean.reshape(-1,1)])
X_test2 = np.hstack([X_test2,test_df.desc_sim_max.reshape(-1,1),test_df.desc_sim_mean.reshape(-1,1)])
print "Concise shape(1): ", X_train2.shape, X_test2.shape

print "[*] SKU characteristic"
X_skuLA_train = np.hstack([train_df.sku_id.apply(lambda x: x.lower().count('a')).reshape(-1,1),train_df.sku_id.apply(lambda x: x.lower().count('l')).reshape(-1,1),train_df.sku_id.apply(lambda x: x.lower().count('la')).reshape(-1,1)])
X_skuLA_test = np.hstack([test_df.sku_id.apply(lambda x: x.lower().count('a')).reshape(-1,1),test_df.sku_id.apply(lambda x: x.lower().count('l')).reshape(-1,1),test_df.sku_id.apply(lambda x: x.lower().count('la')).reshape(-1,1)])
print "SKU stats shape: ", X_skuLA_train.shape, X_skuLA_test.shape

print "[*] TSNE on WV"
title_wv = []
for i, r in train_df.iterrows():
    v = get_sentence_vector(r['title_stem2'])
    title_wv.append(v)
title_wv_test = []
for i, r in test_df.iterrows():
    v = get_sentence_vector(r['title_stem2'])
    title_wv_test.append(v)

X_scaled = np.asarray(title_wv + title_wv_test)
X_tsne = bh_sne(X_scaled)
X_train_title_tsne = X_tsne[:len(train_df)]
X_test_title_tsne = X_tsne[len(train_df):]

# add product type, price, tsne wv
X_train = np.hstack([X_train,X_train_title_tsne])
X_test = np.hstack([X_test,X_test_title_tsne])
X_train2 = np.hstack([X_train2,X_train_title_tsne])
X_test2 = np.hstack([X_test2,X_test_title_tsne])
X_train = np.hstack([X_train,train_df.product_type_le.reshape(-1,1),train_df.price_cvt.reshape(-1,1)])
X_test = np.hstack([X_test,test_df.product_type_le.reshape(-1,1),test_df.price_cvt.reshape(-1,1)])
X_train2 = np.hstack([X_train2,train_df.product_type_le.reshape(-1,1),train_df.price_cvt.reshape(-1,1)])
X_test2 = np.hstack([X_test2,test_df.product_type_le.reshape(-1,1),test_df.price_cvt.reshape(-1,1)])
print "Clarity shape(2): ", X_train.shape, X_test.shape
print "Concise shape(2): ", X_train2.shape, X_test2.shape

print "[*] Desc interaction"
X5_train = []
for i,r in train_df.iterrows():
    X5_train.append(get_token_features(r['title_stem1'],r['desc_stem1']))
X5_train = np.asarray(X5_train)
X5_test = []
for i,r in test_df.iterrows():
    X5_test.append(get_token_features(r['title_stem1'],r['desc_stem1']))
X5_test = np.asarray(X5_test)

X_train = np.hstack([X_train,X5_train])
X_test = np.hstack([X_test,X5_test])
X_train2 = np.hstack([X_train2,X5_train])
X_test2 = np.hstack([X_test2,X5_test])
print "Clarity shape(3): ", X_train.shape, X_test.shape
print "Concise shape(3): ", X_train2.shape, X_test2.shape

X_train = np.hstack([X_train,X_skuLA_train])
X_test = np.hstack([X_test,X_skuLA_test])
X_train2 = np.hstack([X_train2,X_skuLA_train])
X_test2 = np.hstack([X_test2,X_skuLA_test])
print "Clarity shape(4): ", X_train.shape, X_test.shape
print "Concise shape(4): ", X_train2.shape, X_test2.shape

print "[*] Dump to files"
pickle.dump(X_train, open("../features/X_train_{}.p".format(X_train.shape[1]), 'wb'))
pickle.dump(X_test, open("../features/X_test_{}.p".format(X_test.shape[1]), 'wb'))
pickle.dump(X_train2, open("../features/X_train2_{}.p".format(X_train2.shape[1]), 'wb'))
pickle.dump(X_test2, open("../features/X_test2_{}.p".format(X_test2.shape[1]), 'wb'))