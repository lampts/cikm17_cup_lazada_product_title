__author__ = 'lamp'
import numpy as np
import pandas as pd

print("[*] Loading original files")
train_df = pd.read_csv("../data/data_train.csv", encoding='utf-8', names=['country', 'sku_id', 'title', 'cat1', 'cat2', 'cat3', 'description', 'price', 'product_type'])
y_clarity = pd.read_csv("../data/clarity_train.labels", names=['y_clarity'])
y_concise = pd.read_csv("../data/conciseness_train.labels", names=['y_concise'])
train_df['y_clarity'] = y_clarity
train_df['y_concise'] = y_concise
train_df.fillna('', inplace=True)

train_df.to_json("../data/train.json")
valid_df = pd.read_csv("../data/data_valid.csv", encoding='utf-8', names=['country', 'sku_id', 'title', 'cat1', 'cat2', 'cat3', 'description', 'price', 'product_type'])
valid_df.fillna('', inplace=True)
valid_df.to_json("../data/valid.json")

test_df = pd.read_csv("../data/data_test.csv", encoding='utf-8', names=['country', 'sku_id', 'title', 'cat1', 'cat2', 'cat3', 'description', 'price', 'product_type'])
test_df.fillna('', inplace=True)
test_df.to_json("../data/test.json")

print("[*] Merge validation and test frames")
merge_df = pd.concat([valid_df,test_df])
merge_df.to_json("../data/merge/test.json", orient='records')