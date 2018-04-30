package datapreperator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import com.google.common.collect.Lists;
import com.google.gson.Gson;

/**
 * @author harilal
 *  29-Apr-2018
 */
public class TwitterDataPreparator {
	public static void main(String[] args) {
		try {
			String input_file = "resources/Twitter_Formatted.txt";
			String output_file = "resources/Twitter_id.json";
			
			String line = null;
			Set<String> tweetIdSet = new HashSet<String>();
			BufferedReader br = new BufferedReader(new FileReader(input_file));
			BufferedWriter bw = new BufferedWriter(new FileWriter(output_file));
			
			//{"current_ix": 0, "tweet_ids": ["911333326765441025", "890608763698200577"]}
			
			//Map<String, List<String>> eventidTweetIdListMap = Maps.newHashMap();
			
			while ((line = br.readLine()) != null) {
				tweetIdSet.add(line.split("\\s+")[0]);
				//eventidTweetIdListMap.put(line.split("\\s+")[1], Lists.newArrayList());
			}
			
			br.close();
			
			System.out.println("tweetIdSet size : " + tweetIdSet.size());
			//System.out.println("eventidTweetIdListMap size : " + eventidTweetIdListMap.size());
			
			List<String> tweetIdList = Lists.newArrayList(tweetIdSet);
			
			tweetIdSet.clear();
			
			CrawlInputJson inputJson = new CrawlInputJson();
			inputJson.current_ix = 0;
			inputJson.tweet_ids = Lists.newArrayList(tweetIdList);
			
			Gson gson = new Gson();
			bw.write(gson.toJson(inputJson));
			
			/*for (List<String> subList : Iterables.partition(tweetIdList, 200)) {
				CrawlInputJson inputJson = new CrawlInputJson();
				inputJson.current_ix = 0;
				inputJson.tweet_ids = Lists.newArrayList(subList);
				
				Gson gson = new Gson();
				bw.write(gson.toJson(inputJson));
				//break;
			}*/
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
