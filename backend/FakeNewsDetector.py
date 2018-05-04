from datetime import datetime
import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sps
import matplotlib.pyplot as plt
import sklearn.cluster as skcluster


class FakeNewsDetector:
    def __init__(self, dataframe, article_id_col = "article_id", cleaned_timings_col = None):

        self.dataframe = dataframe
        self.article_id_col = article_id_col
        self.cleaned_timings_col = ""
        if cleaned_timings_col is None:
            self.cleaned_timings_col = "cleaned_created_at"
            self.generate_clean_timings()
        else:
            self.cleaned_timings_col = cleaned_timings_col


    def generate_clean_timings(self, source_col = "created_at", date_parse_str = "%a %b %d %H:%M:%S +%f %Y"):
        self.dataframe[self.cleaned_timings_col] = self.dataframe[source_col].apply(lambda x: (datetime.datetime.strptime(x,date_parse_str) - datetime.datetime.utcfromtimestamp(0)).total_seconds() * 1000)

    def get_time_article(self, id_article):
        arr_times = self.dataframe[self.dataframe[self.article_id_col] == id_article][self.cleaned_timings_col].values
        min_date = min(arr_times)
        arr_times = arr_times - min_date
        return arr_times


    def plot_values(self, article_id):

        arr_values = self.get_time_article(article_id)
        arr_ones = np.ones(len(arr_values))
        plt.figure(figsize=(20, 3))
        plt.plot(arr_values, arr_ones, 'ro', ms=4)
        # plt.axis([0, 1000000000])
        axes = plt.gca()
        axes.set_xlim([0, 80000000000])
        plt.show()


    def cluster_sequence (self, sequence, eps = 86400000, min_samples = 1):
        min_date = min(sequence)
        sequence = sequence - min_date
        index_row = np.arange(len(sequence))
        emitted_matrix = sps.coo_matrix((sequence, (index_row, np.zeros(len(sequence)))), shape=(max(index_row) + 1, 2))
        dbscan = skcluster.DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=-1)
        labels = dbscan.fit_predict(emitted_matrix, y=None, sample_weight=None)
        return labels

    def cluster_article(self, id_article, eps = 86400000, min_samples = 1):
        emitted_dates = self.dataframe[self.dataframe[self.article_id_col] == id_article][self.cleaned_timings_col].values
        return self.cluster_sequence(emitted_dates, eps = eps, min_samples = min_samples)

    def cluster_table(self, new_col = "cluster", eps = 86400000, min_samples = 1):
        self.dataframe[new_col] = self.dataframe.groupby("article_id")["cleaned_created_at"].transform(lambda x: self.cluster_sequence(x, eps = eps, min_samples = min_samples))

    def calculate_delta_t(self, delta_t_col = "delta_t"):
        for i in range(max(self.dataframe[self.article_id_col])+1):
            if i%50 == 0:
                print(i)
            sub_c = self.dataframe[self.dataframe[self.article_id_col] == i]
            for j in range (int(max(sub_c["cluster"])+1)):
                current_t = min(sub_c[sub_c["cluster"] == j]["cleaned_created_at"])
                result = 0
                if j > 0:
                    last_t = max(sub_c[sub_c["cluster"] == j-1]["cleaned_created_at"])
                    result = abs(current_t - last_t)
                self.dataframe.loc[((self.dataframe[self.article_id_col] == i) & (self.dataframe["cluster"] == j)), delta_t_col] = result