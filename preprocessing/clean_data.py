"""
This file cleans the genres of the books.
A dictionary is generated and genre titles are shorten to combine them.
e.g. historical fiction and science fiction => fiction
with this, genres are one-hot encoded (data_genres.csv)
"""

import pandas as pd


# data = pd.read_csv('../data/clean_data_nn.csv')
# data = data.drop(columns=['author','description'])
# data = data.where(data['genres'] != '[]')
# data = data.dropna()
# data.to_csv("./data/clean_dataJJ.csv", index=False)

def clean_alt_list(list_):
    list_ = list_.replace(', ', '","')
    list_ = list_.replace('[', '["')
    list_ = list_.replace(']', '"]')
    return list_



def isEnglish(l):
    new_l = []
    for s in l:
        if s.isascii():
            new_l.append(s)
    return new_l


if __name__ == "__main__":
    data = pd.read_csv("../data/clean_dataJJ.csv")
    data['genres'] = data['genres'].apply(clean_alt_list)
    data['genres'] = data['genres'].apply(eval)
    genres = data['genres']
    # genres.to_csv('./genres.csv', index=False, header=None)

    genres_dict = {}
    for i in genres:
        for j in i:
            j = j.replace("'", '')
            if len(j.split()) > 1:
                j = j.split()[-1]
            if j not in genres_dict:
                genres_dict[j] = 1
            else:
                genres_dict[j] += 1
    sorted_genres = dict(sorted(genres_dict.items(), key=lambda x: x[1]))
    series = pd.Series(sorted_genres)
    series = series.iloc[::-1]
    # bigger 400
    series = series[series > 1000]
    genres_list = series.index.to_list()
    genres_df = pd.DataFrame(columns=genres_list)
    genres_df["data"] = genres

    for idx, i in enumerate(genres_df["data"]):
        for j in i:
            j = j.replace("'", '')
            if len(j.split()) > 1:
                j = j.split()[-1]
            if j in genres_df.columns:
                genres_df[j][idx] = 1
    print(genres_df)
    for c in genres_list:
        genres_df[c] = genres_df[c].fillna(0)
    genres_df_drop = genres_df.drop(columns='data')
    data = data.join(genres_df_drop)
    data = data.drop(columns='genres')
    data['title'] = data['title'].str.replace(',', '')
    data['title'] = data['title'].str.replace('"', '')
    # extracts the genres for each book and saves it as a csv.
    data.to_csv("data_genres.csv", index=False)
