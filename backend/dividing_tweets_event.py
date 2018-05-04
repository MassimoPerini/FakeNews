import pandas as pd
from random import shuffle


def write_csv_header(file, df):
    index = 0
    for name in list(df):
        file.write(name)
        index += 1
        if index < len(list(df)):
            file.write(",")
    file.write("\n")





INPUT_FILE = "complete_table_end.csv"
OUTPUT_FILE_TEST = "test_dataset.csv"
OUTPUT_FILE_TRAINING = "training_dataset.csv"
PERCENTAGE_OF_TRAINING_SET = 0.8
TWEETS_PER_EVENT = 5


input_file = pd.read_csv(INPUT_FILE, sep=";", lineterminator="\n")
testing_file = open(OUTPUT_FILE_TEST, "w")

#input_file.to_csv(testing_file)

testing_file = open(OUTPUT_FILE_TEST, "w")
training_file = open(OUTPUT_FILE_TRAINING, "w")

tweets_events = []
events_array = []

for index in range(0, len(input_file)):
    #print(index, len(input_file))
    if index < len(input_file) - 1:
        current_id_article = input_file["article_id"][index]
        tweets_events.append(index)
        if current_id_article != input_file["article_id"][index + 1]:
            if len(tweets_events) > TWEETS_PER_EVENT:
                shuffle(tweets_events)
            events_array.append(tweets_events[:TWEETS_PER_EVENT])
            tweets_events = []

shuffle(events_array)
number_samples_training = round(PERCENTAGE_OF_TRAINING_SET * len(events_array))

training_events = events_array[:number_samples_training]
testing_events = events_array[number_samples_training:]

training_file = open(OUTPUT_FILE_TRAINING, "w")
testing_file = open(OUTPUT_FILE_TEST, "w")

print("Writing on training file")
write_csv_header(training_file, input_file)
for event in training_events:
    for tweet in event:
        row =  input_file.iloc[tweet]
        pd.DataFrame(row).T.to_csv(training_file, header=False, index=False)


print("Writing on test file...")
write_csv_header(testing_file, input_file)
for event in testing_events:
    for tweet in event:
        row = input_file.iloc[tweet]
        pd.DataFrame(row).T.to_csv(testing_file, header=False, index=False)


