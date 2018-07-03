import pandas as pd
import numpy as np

FILE = "10_tweets_formatted_tweets.csv"
SEP = "|"
OUTPUT_FILE = "./1_tweet_formatted_tweets.csv"

csv_file = pd.read_csv(FILE, sep=SEP)
row_number = csv_file.shape[0]
unique_label_events = list(csv_file["event_id"].unique())
vector_indices = np.zeros(len(unique_label_events))
final_array_index = 0

for row_index in range(row_number):
    print(row_index, row_number)
    current_label_event = csv_file.iloc[row_index][2]
    if current_label_event in unique_label_events:
        unique_label_events.remove(current_label_event)
        vector_indices[final_array_index] = row_index
        final_array_index += 1

final_events = csv_file.loc[vector_indices]
final_events.to_csv(OUTPUT_FILE, sep=SEP)



