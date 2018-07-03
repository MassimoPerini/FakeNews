import sys
sys.path.append("/usr/local/lib/python3.6/site-packages")
from py4j.java_gateway import JavaGateway
import json
import pandas as pd

TEST_FILE = "dataset/final_test_dataset_5.csv"
FILE_TO_SEARCH = "dataset/final_test_dataset_5.csv"
OUTPUT_FILE = "dataset/FastText_prediction.json"

with open(TEST_FILE) as test_file:
    complete_file = pd.read_csv(test_file, sep="|")


#complete_file.set_index("id", inplace=True)

gateway = JavaGateway()

output_file = open(OUTPUT_FILE, "w")

to_dump = []
for index in range(len(complete_file)):
    target_tweet_id = complete_file["id"][index]
    #print(target_tweet_id)
    text_to_analyze = complete_file.loc[complete_file["id"] == int(target_tweet_id)]
    text_to_analyze = text_to_analyze["url_content"].tolist()[0]
    prediction = gateway.entry_point.getPrediction(text_to_analyze)
    is_fake = complete_file.loc[complete_file["id"] == int(target_tweet_id)]
    #print(is_fake)
    is_fake = is_fake["is_fake"].tolist()[0]
    #print(is_fake)
    entry = {'tweet_id': str(target_tweet_id), 'predictedId': str(prediction), 'actual_fake': str(is_fake)}
    to_dump.append(entry)

json.dump(to_dump, output_file)