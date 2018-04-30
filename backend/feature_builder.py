
import json

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


with open('formatted_tweets.json') as json_data:
    formatted_tweets = json.load(json_data)

with open('response.json') as json_data:
    tweets_analysis = json.load(json_data)

tweets_csv = open("tweets_dataset.csv", "w")
tweets_csv.write("tweet_id, fake, retweet_on_favorite, joy, sadness, anger, fear, disgust, sentiment\n")

tweet_formatted_index = 0

for tweet_analysis_index in range(len(tweets_analysis["tweets"])):

    print(tweet_analysis_index, len(tweets_analysis["tweets"]))
    while True:
        target_tweet_formatted_id = formatted_tweets["tweets"][tweet_formatted_index]["id"]
        target_tweet_analysis_id = tweets_analysis["tweets"][tweet_analysis_index]["tweet_id"]
        if str(target_tweet_formatted_id) == str(target_tweet_analysis_id):
            break
        else:
            tweet_formatted_index += 1

    target_tweet_analysis = tweets_analysis["tweets"][tweet_analysis_index]
    target_tweet_formatted = formatted_tweets["tweets"][tweet_formatted_index]

    tweets_csv.write(str(target_tweet_analysis["tweet_id"]))
    tweets_csv.write(", ")

    tweets_csv.write(str(target_tweet_formatted["is_fake"]))
    tweets_csv.write(", ")

    number_favorite = target_tweet_formatted["favorite_count"]
    number_retweet = target_tweet_formatted["retweet_count"]
    if number_retweet != 0:
        retweet_on_favorite = number_favorite/number_retweet
    else:
        retweet_on_favorite = 0
    tweets_csv.write(str(retweet_on_favorite))
    tweets_csv.write(", ")

    total_joy = average_emotion(target_tweet_analysis, "joy")
    tweets_csv.write(str(total_joy))
    tweets_csv.write(", ")

    total_sadness = average_emotion(target_tweet_analysis, "sadness")
    tweets_csv.write(str(total_sadness))
    tweets_csv.write(", ")

    total_anger = average_emotion(target_tweet_analysis, "anger")
    tweets_csv.write(str(total_anger))
    tweets_csv.write(", ")

    total_fear = average_emotion(target_tweet_analysis, "fear")
    tweets_csv.write(str(total_fear))
    tweets_csv.write(", ")

    total_disgust = average_emotion(target_tweet_analysis, "disgust")
    tweets_csv.write(str(total_disgust))
    tweets_csv.write(", ")

    total_sentiment = average_sentiment(target_tweet_analysis)
    tweets_csv.write(str(total_sentiment))
    tweets_csv.write("\n")

    tweet_formatted_index += 1

tweets_csv.close()
