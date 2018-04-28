package datapreperator;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

public class DataSplitter {
	public static void main(String[] args) {
		try {
			String input_file = "/home/harilal/Documents/Polimi_Academics/Second_Semester/Designing_Digital_Business_Innovation_Lab/Datasets/rumdect/Twitter.txt";
			String output_file = "/home/harilal/Documents/Polimi_Academics/Second_Semester/Designing_Digital_Business_Innovation_Lab/Datasets/rumdect/Twitter_Formatted.txt";

			String line = null;
			BufferedReader br = new BufferedReader(new FileReader(input_file));
			BufferedWriter bw = new BufferedWriter(new FileWriter(output_file));

			int count = 0;
			String[] splitArray = null;
			while ((line = br.readLine()) != null) {
				//System.out.println(line);
				splitArray = line.split("\\s+", 3);
				/*System.out.println(splitArray[0]);
				System.out.println(splitArray[1]);
				System.out.println(splitArray[2]);*/

				for (String tweet_id : splitArray[2].split("\\s+")) {
					bw.write(tweet_id + " " + splitArray[0].split("\\:")[1] + " " + splitArray[1].split("\\:")[1] + "\n");
				}
			}

			br.close();
			bw.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}