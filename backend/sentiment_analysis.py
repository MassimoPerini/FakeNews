from __future__ import print_function
import json
import pandas as pd
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions

#If the input file is a JSON or a csv file.
#One of them (only one) must me True

JSON = False
CSV = True
SEP = '|'           #separator for the csv file
FILE = '10_tweets_formatted_tweets.csv'

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='0bceda96-a518-4425-99ae-60f034b16963',
    password='3YFWZ4kFEmeL')

if JSON:
    with open(FILE) as json_data:
        json_file = json.load(json_data)

elif CSV:
    csv_file = pd.read_csv(FILE, sep=SEP)

else:
    raise Exception("Both JSON and CSV are false")

file  = open("./response.json", "w")
file.write('{\n\t"tweets": [')
#max_range = len(d["tweets"]) #all tweets
min_range = 0
max_range = 300
for index in range(min_range, max_range):

    print(index, max_range)

    if JSON:
        tweet_id = json_file["tweets"][index]["tweet_text"]
        tweet_text = json_file["tweets"][index]["id"]
    if CSV:
        tweet_id = csv_file["id"][index]
        tweet_text = csv_file["tweet_text"][index]

    try:
        response = natural_language_understanding.analyze(
            text= tweet_text,
            features=Features(entities=EntitiesOptions(), keywords=KeywordsOptions(sentiment=True, emotion=True)))
        file.write('\n\t\t{\n\t\t"tweet_id": ')
        file.write(str(tweet_id))
        file.write(',\n\t\t"tweet_analysis":\n\t\t\t')
        file.write(json.dumps(response))
        file.write('\n\t\t}')
        if index < max_range - 1:
            file.write(',')
    except:
        continue

file.write('\n\t]\n}')
file.close()