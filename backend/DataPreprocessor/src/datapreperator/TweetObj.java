package datapreperator;

import java.util.List;

/**
 * @author harilal
 *  29-Apr-2018
 */
public class TweetObj {
	String created_at;
	String id;
	String tweet_text;
	String user_id;
	String in_reply_to_status_id;//"NULL" if no such value
	List<String> hash_tags;
	List<String> user_mentions;
	Integer retweet_count;
	Integer favorite_count;
	Boolean possibly_sensitive;//"NULL" if no value
}
