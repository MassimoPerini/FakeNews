package datapreperator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * @author harilal
 *  29-Apr-2018
 */
public class DataPreprocessor {
	public static void main(String[] args) {
		try {
			String input_file = "resources/crawled_tweets.json";
			String output_file = "resources/formatted_tweets.json";

			BufferedReader br = new BufferedReader(new FileReader(input_file));
			BufferedWriter bw = new BufferedWriter(new FileWriter(output_file));

			String line = null, formattedJson = null;

			while ((line = br.readLine()) != null) {
				formattedJson = getFormattedTweetJson(line);

				if (formattedJson != null) {
					bw.write(formattedJson + "\n");
				}
			}

			br.close();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static String getFormattedTweetJson(String json) {
		String formattedJson = null;
		try {
			JsonParser parser = new JsonParser();

			JsonElement jsonTree = parser.parse(json);

			if (jsonTree.isJsonObject()) {
				JsonObject tweetJsonObject = jsonTree.getAsJsonObject();

				JsonElement tweetCreatedAt  = tweetJsonObject.get("created_at");
				JsonElement tweetId  = tweetJsonObject.get("id_str");
				JsonElement tweetText  = tweetJsonObject.get("full_text");
				JsonElement tweetRetweetCount  = tweetJsonObject.get("retweet_count");
				JsonElement tweetFavouriteCount  = tweetJsonObject.get("favorite_count");
				JsonElement tweetPossiblySensitive  = tweetJsonObject.get("possibly_sensitive");
				JsonElement tweetInReplyToStatusId = tweetJsonObject.get("in_reply_to_status_id_str");;

				TweetObj tweetObj = new TweetObj();

				tweetObj.created_at = tweetCreatedAt.getAsString();
				tweetObj.id = tweetId.getAsString();
				tweetObj.tweet_text = tweetText.getAsString();
				tweetObj.retweet_count = tweetRetweetCount.getAsInt();
				tweetObj.favorite_count = tweetFavouriteCount.getAsInt();

				if (!tweetInReplyToStatusId.isJsonNull()) {
					tweetObj.in_reply_to_status_id = tweetInReplyToStatusId.getAsString();
				} else {
					tweetObj.in_reply_to_status_id = "NULL";
				}

				if (tweetPossiblySensitive != null) {
					tweetObj.possibly_sensitive = tweetPossiblySensitive.getAsBoolean();
				} else {
					tweetObj.possibly_sensitive = false;
				}


				tweetObj.hash_tags = Lists.newArrayList();
				tweetObj.user_mentions = Lists.newArrayList();

				JsonElement entitiesElement = tweetJsonObject.get("entities");

				if (entitiesElement.isJsonObject()) {
					JsonObject entitiesJsonObject = entitiesElement.getAsJsonObject();

					JsonElement hashTagElement = entitiesJsonObject.get("hashtags");

					JsonArray hashTagJsonArray = hashTagElement.getAsJsonArray();

					if (hashTagJsonArray.size() > 0) {
						for (JsonElement jsonElement : hashTagJsonArray) {
							JsonObject hashTagJsonObject = jsonElement.getAsJsonObject();

							JsonElement hashTag = hashTagJsonObject.get("text");

							tweetObj.hash_tags.add(hashTag.getAsString());
						}
					}


					JsonElement userMentionElement = entitiesJsonObject.get("user_mentions");

					JsonArray userMentionJsonArray = userMentionElement.getAsJsonArray();

					if (userMentionJsonArray.size() > 0) {
						for (JsonElement jsonElement : userMentionJsonArray) {
							JsonObject userMentionJsonObject = jsonElement.getAsJsonObject();

							JsonElement hashTag = userMentionJsonObject.get("id_str");

							tweetObj.user_mentions.add(hashTag.getAsString());
						}
					}
				}

				JsonElement userElement = tweetJsonObject.get("user");

				UserObj userObj = new UserObj();

				if (userElement.isJsonObject()) {
					JsonObject userJsonObject = userElement.getAsJsonObject();

					JsonElement userId = userJsonObject.get("id_str");
					JsonElement userName = userJsonObject.get("name");
					JsonElement userScreeName = userJsonObject.get("screen_name");
					JsonElement userLocation = userJsonObject.get("location");
					JsonElement userDescription = userJsonObject.get("description");
					JsonElement userIsProtected = userJsonObject.get("protected");
					JsonElement userFollowersCount = userJsonObject.get("followers_count");
					JsonElement userFriendsCount = userJsonObject.get("friends_count");
					JsonElement userListedCount = userJsonObject.get("listed_count");
					JsonElement userCreatedAt = userJsonObject.get("created_at");
					JsonElement userFavouritesCount = userJsonObject.get("favourites_count");
					JsonElement userIsVerified = userJsonObject.get("verified");
					JsonElement userStatusesCount = userJsonObject.get("statuses_count");
					JsonElement userTimeZone = userJsonObject.get("time_zone");

					userObj.id = userId.getAsString();
					userObj.name = userName.getAsString();
					userObj.screen_name = userScreeName.getAsString();
					userObj.location = userLocation.getAsString();
					userObj.description = userDescription.getAsString();
					userObj.is_protected = userIsProtected.getAsBoolean();
					userObj.followers_count = userFollowersCount.getAsInt();
					userObj.friends_count = userFriendsCount.getAsInt();
					userObj.listed_count = userListedCount.getAsInt();
					userObj.created_at = userCreatedAt.getAsString();
					userObj.favourites_count = userFavouritesCount.getAsInt();
					userObj.is_verified = userIsVerified.getAsBoolean();
					userObj.statuses_count = userStatusesCount.getAsInt();
					if (userTimeZone != null && !userTimeZone.isJsonNull()) {
						userObj.time_zone = userTimeZone.getAsString();
					} else {
						userObj.time_zone = "NULL";
					}
				}

				tweetObj.user_id = userObj.id;

				Gson newGsonObj = new Gson();
				formattedJson = newGsonObj.toJson(tweetObj);
			}
			return formattedJson;
		} catch (Exception e) {
			e.printStackTrace();
			//return null;
		}
		return formattedJson;
	}
}
