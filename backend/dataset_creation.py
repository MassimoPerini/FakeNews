import pandas as pd
import json



JSON = False
CSV = True
SEP = '|'           #separator for the csv file
FORMATTED_INPUT_FILE = 'dataset/final_test_dataset_5_threshold_1500.csv'
SENTIMENT_FILE = 'sentiment_analysis/final_test_dataset_5_threshold_1500_sentiment.json'
OUTPUT_FILE = "sentiment_analysis/final_test_dataset_5_threshold_1500.csv"
CLUSTER_FILE = ""
CLUSTER = False
ELIMINATE_OUTLIERS = True




def average_emotion(target_tweet_analysis, target_emotion):

    number_keywords = len(target_tweet_analysis["tweet_analysis"]["keywords"])
    total_emotion = 0.0
    for keyword_index in range(number_keywords):
        target_keyword = target_tweet_analysis["tweet_analysis"]["keywords"][keyword_index]
        try:
            total_emotion += target_keyword["emotion"][target_emotion]*target_keyword["relevance"]
        except KeyError:
            total_emotion += 0

    return total_emotion


def average_sentiment(target_tweet_analysis):

    number_keywords = len(target_tweet_analysis["tweet_analysis"]["keywords"])
    total_sentiment = 0.0
    for keyword_index in range(number_keywords):
        target_keyword = target_tweet_analysis["tweet_analysis"]["keywords"][keyword_index]
        try:
            total_sentiment += target_keyword["sentiment"]["score"]*target_keyword["relevance"]
        except KeyError:
           total_sentiment += 0

    return total_sentiment

#If the input file is a JSON or a csv file.
#One of them (only one) must me True

if JSON:
    with open(FORMATTED_INPUT_FILE) as json_data:
        formatted_tweets = json.load(json_data)

elif CSV:
    formatted_tweets = pd.read_csv(FORMATTED_INPUT_FILE, sep=SEP)
    if CLUSTER:
        df_file = pd.read_csv(CLUSTER_FILE)
else:
    raise Exception("Both JSON and CSV are false")


with open(SENTIMENT_FILE) as json_data:
    tweets_analysis = json.load(json_data)

tweets_csv = open(OUTPUT_FILE, "w")
tweets_csv.write("tweet_id,fake,joy,sadness,anger,fear,disgust,sentiment")

if CLUSTER:
    tweets_csv.write(",delta_t,cluster")

tweets_csv.write("\n")

if JSON:
    number_tweets = len(formatted_tweets["tweets"])
elif CSV:
    number_tweets = len(formatted_tweets)

list_id_formatted_tweets = formatted_tweets["id"].tolist()

for tweet_analysis_index in range(len(tweets_analysis["tweets"])):

    print(tweet_analysis_index, len(tweets_analysis["tweets"]))

    analysis_tweet_id = tweets_analysis["tweets"][tweet_analysis_index]["tweet_id"]
    if JSON:
        tweet_formatted_index = formatted_tweets["tweets"]["id"].index(analysis_tweet_id)
    elif CSV:
        tweet_formatted_index = list_id_formatted_tweets.index(analysis_tweet_id)

    target_tweet_analysis = tweets_analysis["tweets"][tweet_analysis_index]

    total_joy = average_emotion(target_tweet_analysis, "joy")
    total_sadness = average_emotion(target_tweet_analysis, "sadness")
    total_anger = average_emotion(target_tweet_analysis, "anger")
    total_fear = average_emotion(target_tweet_analysis, "fear")
    total_disgust = average_emotion(target_tweet_analysis, "disgust")
    total_sentiment = average_sentiment(target_tweet_analysis)

    if ELIMINATE_OUTLIERS:
        if total_joy == 0 and total_sadness == 0 and \
                total_anger == 0 and total_fear == 0 and total_disgust == 0:
            continue


    tweets_csv.write(str(target_tweet_analysis["tweet_id"]))
    tweets_csv.write(SEP)

    if JSON:
        tweets_csv.write(str(formatted_tweets["tweets"][tweet_formatted_index]["is_fake"]))

    elif CSV:
        tweets_csv.write(str(formatted_tweets["is_fake"].iloc[tweet_formatted_index]))

    tweets_csv.write(SEP)

    tweets_csv.write(str(total_joy))
    tweets_csv.write(SEP)

    tweets_csv.write(str(total_sadness))
    tweets_csv.write(SEP)

    tweets_csv.write(str(total_anger))
    tweets_csv.write(SEP)

    tweets_csv.write(str(total_fear))
    tweets_csv.write(SEP)

    tweets_csv.write(str(total_disgust))
    tweets_csv.write(SEP)

    tweets_csv.write(str(total_sentiment))
    tweets_csv.write(SEP)

    if CLUSTER:
        tweets_csv.write(str(df_file["delta_t"][tweet_analysis_index]))
        tweets_csv.write(SEP)

        tweets_csv.write(str(df_file["cluster"][tweet_analysis_index]))
    tweets_csv.write("\n")

    tweet_formatted_index += 1

tweets_csv.close()
