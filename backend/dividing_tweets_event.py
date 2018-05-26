import pandas as pd
from random import shuffle
import numpy as np

INPUT_FILE = "csv_dataset_with_articles_and_locations.csv"
SEP = "|"
OUTPUT_FILE_TEST = "dataset/final_test_dataset"
OUTPUT_FILE_TRAINING = "dataset/final_training_dataset"
train_test_split = 0.8
TWEETS_PER_EVENT = 20
#Threshold on the minimum number of characters of the articles. Put 0 to don't use the threshold
CHARACTERS_THRESHOLD = 1500

OUTPUT_FILE_TEST += "_" + str(TWEETS_PER_EVENT) + "_" + "threshold_" + str(CHARACTERS_THRESHOLD) + ".csv"
OUTPUT_FILE_TRAINING += "_" + str(TWEETS_PER_EVENT) + "_" + "threshold_" + str(CHARACTERS_THRESHOLD) + ".csv"

def write_csv_header(file, df):
    index = 0
    for name in list(df):
        file.write(name)
        index += 1
        if index < len(list(df)):
            file.write(SEP)
    file.write("\n")


input_file = pd.read_csv(INPUT_FILE, sep=SEP)

#input_file.to_csv(testing_file)


training_file = open(OUTPUT_FILE_TRAINING, "w")
testing_file = open(OUTPUT_FILE_TEST, "w")

number_tweets = len(input_file["event_id"])

unique_events_array = []

for index in range(number_tweets):
    #print(index, number_tweets)
    tweet_event = input_file["event_id"][index]
    if tweet_event not in unique_events_array:
        unique_events_array.append(tweet_event)




number_unique_events = len(unique_events_array)

index_vector = np.arange(0, number_unique_events, 1)
event_dictonary = dict(zip(unique_events_array, index_vector))
tweets_events=[[] for i in np.arange(number_unique_events)]


for index in range(number_tweets):
    tweet_event = input_file["event_id"][index]
    target_index = event_dictonary.get(tweet_event)
    tweets_events[target_index].append(index)

#train_mask = np.random.choice([True, False], number_unique_events, p=[train_test_split, 1-train_test_split])
#test_mask = np.logical_not(train_mask)

#training_events = tweets_events[train_mask]
#testing_events = tweets_events[test_mask]

np.random.shuffle(tweets_events)
index_for_splitting = round(train_test_split*number_unique_events)
training_events = tweets_events[:index_for_splitting]
testing_events = tweets_events[index_for_splitting:]


for index in range(len(training_events)):
    event = training_events[index]
    shuffle(event)
    if len(event) >= TWEETS_PER_EVENT:
        if CHARACTERS_THRESHOLD > 0:
            indices_for_threshold = []
            missing_tweets = TWEETS_PER_EVENT
            for tweet_index in range(len(event)):
                tweet = event[tweet_index]
                article_length = len(input_file.iloc[tweet]["url_content"])
                if article_length >= CHARACTERS_THRESHOLD:
                    indices_for_threshold.append(tweet_index)
                    missing_tweets -= 1
                    if missing_tweets == 0:
                        break

            if missing_tweets > 0:
                for tweet_index in range(len(event)):
                    if tweet_index not in indices_for_threshold:
                        indices_for_threshold.append(tweet_index)
                        missing_tweets -= 1
                        if missing_tweets == 0:
                            break
        else:
            indices_for_threshold = np.arange(TWEETS_PER_EVENT)

        training_events[index] = [event[i] for i in indices_for_threshold]
    else:
        training_events[index] = event

for index in range(len(testing_events)):
    event = testing_events[index]
    shuffle(event)
    if len(event) >= TWEETS_PER_EVENT:
        if CHARACTERS_THRESHOLD > 0:
            indices_for_threshold = []
            missing_tweets = TWEETS_PER_EVENT
            for tweet_index in range(len(event)):
                tweet = event[tweet_index]
                article_length = len(input_file.iloc[tweet_index]["url_content"])
                if article_length >= CHARACTERS_THRESHOLD:
                    indices_for_threshold.append(tweet_index)
                    missing_tweets -= 1
                    if missing_tweets == 0:
                        break

            if missing_tweets > 0:
                for tweet_index in range(len(event)):
                    if tweet_index not in indices_for_threshold:
                        indices_for_threshold.append(tweet_index)
                        missing_tweets -= 1
                        if missing_tweets == 0:
                            break
        else:
            indices_for_threshold = np.arange(TWEETS_PER_EVENT)

        testing_events[index] = [event[i] for i in indices_for_threshold]

    else:
        testing_events[index] = event


print("Writing on training file...")
write_csv_header(training_file, input_file)
event_counter = 0
total_tweet_counter = 0
for event in training_events:
    for tweet in event:
        row =  input_file.iloc[tweet]
        pd.DataFrame(row).T.to_csv(training_file, header=False, index=False, sep=SEP)
        total_tweet_counter += 1
    event_counter += 1

training_file.close()

print("TRAINING FILE")
print("\nNumber of events: ", event_counter)
print("Number of total tweets: ", total_tweet_counter)

print("Writing on test file...")
write_csv_header(testing_file, input_file)
event_counter = 0
total_tweet_counter = 0
for event in testing_events:
    for tweet in event:
        row = input_file.iloc[tweet]
        pd.DataFrame(row).T.to_csv(testing_file, header=False, index=False, sep=SEP)
        total_tweet_counter += 1
    event_counter += 1

testing_file.close()
print("\nTESTING FILE")
print("\nNumber of events: ", event_counter)
print("Number of total tweets: ", total_tweet_counter)
