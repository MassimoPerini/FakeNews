from __future__ import print_function
import json
import pandas as pd
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
import sys

#If the input file is a JSON or a csv file.
#One of them (only one) must me True

SEP = '|'           #separator for the csv file
INPUT_FILE = 'dataset/final_test_dataset_5.csv'
OUTPUT_FILE = 'sentiment_analysis/final_test_dataset_5_sentiment.json'

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='cfe82a94-0e08-4b2c-89ce-7a7e2f2a5855',
    password='rOmhJWBQrZpP')

df = pd.read_csv(INPUT_FILE, sep=SEP)

file = open(OUTPUT_FILE, "w")
file.write('{\n\t"tweets": [')
#max_range = len(d["tweets"]) #all tweets

starting_tweet = 0
finishing_tweet = len(df)

#OUTPUT_FILE = OUTPUT_FILE + str(starting_tweet) + "_" + str(finishing_tweet) + ".json"

for index in range(starting_tweet, finishing_tweet):

    print(index, finishing_tweet)
    tweet_id = df["id"][index]
    tweet_text = df["url_content"][index]

    try:
        response = natural_language_understanding.analyze(
            text= tweet_text,
            features=Features(entities=EntitiesOptions(), keywords=KeywordsOptions(sentiment=True, emotion=True)))
        file.write('\n\t\t{\n\t\t"tweet_id": ')
        file.write(str(tweet_id))
        file.write(',\n\t\t"tweet_analysis":\n\t\t\t')
        file.write(json.dumps(response))
        file.write('\n\t\t}')
        if index < finishing_tweet - 1:
            file.write(',')
    except Exception as e:
        print(e)
        continue

file.write('\n\t]\n}')
file.close()
