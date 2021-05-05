#!/usr/bin/env python3
# importing libraries
from OneHot import *
from DataClean import *
from DashboardController import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    #Fetches our csv files that we need to sort
    data_set_movies = pd.read_csv(r"datasets\IMDbmovies.csv", low_memory=False)
    data_set_ratings = pd.read_csv(r"datasets\IMDbratings.csv", low_memory=False)
    #List of the columns we want to keep
    keepList = ['genre','reviews_from_critics','avg_vote','year','duration']

    clean_data = DataClean(data_set_movies.head(1000))
    cleaned_data = clean_data.full_clean(data_set_ratings,keepList)

    onehot = OneHot(cleaned_data,'genre')
    encoded_df = onehot.encode()

    collected_df = pd.merge(encoded_df,cleaned_data,left_index=True,right_index=True)
    del collected_df['genre']

    plt.figure(figsize=(20, 10))
    cor = collected_df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()

    dashboard_controller = DashboardController(collected_df)
    dashboard_controller.dash_application()

if __name__ == "__main__":
    main()