# Required packages

(1) In order to reproduce, we need to install following packages
- numpy
- scipy
- pandas
- tensorflow
- keras
- sklearn
- nltk
- spacy
- fuzzywuzzy
- tsne
- xgboost

(2) Then run the scripts inside code folder in the sequential step to extract features, build models and transfer features.

(3) The project structure:
.
├── clean
│   ├── 10_train_concise.py
│   ├── 11_transfer_learning.py
│   ├── 1_extract_dataframe.py
│   ├── 2_build_features.py
│   ├── 3_generate_sequences.py
│   ├── 4_generate_sku_sequence.py
│   ├── 5_generate_new_features.py
│   ├── 6_generate_graph_title.py
│   ├── 7_generate_graph_desc.py
│   ├── 8_generate_tsne_desc.py
│   ├── 9_train_clarity.py
│   └── README.txt
├── data
│   ├── clarity_train.labels
│   ├── clarity_valid.predict
│   ├── conciseness_train.labels
│   ├── conciseness_valid.predict
│   ├── data_test.csv
│   ├── data_train.csv
│   ├── data_valid.csv
│   ├── merge
│   │   └── test.json
│   ├── test.json
│   ├── train.json
│   └── valid.json
├── features
│   └── emb
│       ├── new_laz_glove_25K.mat
│       └── new_word2ix.pkl
├── log
├── models
│   
└── submit
