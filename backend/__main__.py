
import json
import numpy as np
import scipy as scs
import pandas as pd
from pandas.io.json import json_normalize

from FakeNewsDetector import FakeNewsDetector

print("Starting...")
'''
data = json.load(open('dataset/formatted_tweets.json'))
data = data["tweets"]
json_normalize(data)
df = pd.DataFrame(data)

list_id = sorted(set(df["id"]))
df["id"] = df["id"].apply(lambda x: list_id.index(x))

print("done...")

list_id = sorted(set(df["user_id"]))
df["user_id"] = df["user_id"].apply(lambda x: list_id.index(x))

print("done...")


df.to_csv("cleaned_dataset", index=False)


print(df.head())
'''

df = pd.read_csv("dataset/complete_table_end.csv", lineterminator='\n', sep=";")
fake_news = FakeNewsDetector(df, article_id_col = "article_id")
print("start clustering...")
fake_news.cluster_table()
print("start computation of delta t")
fake_news.calculate_delta_t()
print(df.head(20))