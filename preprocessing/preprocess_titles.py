"""
This file preprocesses the title in the cleaned data with encoded genres.
"""

import pandas as pd
import re

if __name__ == "__main__":
    data = pd.read_csv("../data/data_genres.csv")
    # Remove "The "
    data['title'] = data['title'].str.split(':').str[0]
    data['title'] = data['title'].str.split('?').str[0]
    data['title'] = data['title'].str.split('(').str[0]
    data['title'] = data['title'].str.split('–').str[0]
    data['title'] = data['title'].str.split('/').str[0]
    data['title'] = data['title'].str.split('[').str[0]
    data['title'] = data['title'].str.replace("'", '')
    data['title'] = data['title'].str.replace("!", ' ')
    data['title'] = data['title'].str.replace(".", '')
    data['title'] = data['title'].str.replace("’", '')
    data['title'] = data['title'].str.replace("´", '')
    data['title'] = data['title'].str.replace("Á", 'A')
    data['title'] = data['title'].str.replace("À", 'A')
    data['title'] = data['title'].str.replace("á", 'a')
    data['title'] = data['title'].str.replace("à", 'a')
    data['title'] = data['title'].str.replace("ä", 'a')
    data['title'] = data['title'].str.replace("é", 'e')
    data['title'] = data['title'].str.replace("è", 'e')
    data['title'] = data['title'].str.replace("ë", 'e')
    data['title'] = data['title'].str.replace("ö", 'o')
    data['title'] = data['title'].str.replace("ò", 'o')
    data['title'] = data['title'].str.replace("ó", 'o')
    data['title'] = data['title'].str.replace("ü", 'u')
    data['title'] = data['title'].str.replace("ú", 'u')
    data['title'] = data['title'].str.replace("½", '')
    data['title'] = data['title'].str.replace("—", ' ')
    data['title'] = data['title'].str.replace("-", ' ')
    data['title'] = data['title'].str.replace("#", '')
    data['title'] = data['title'].str.replace("&", 'and')
    data['title'] = data['title'].str.replace("[0-9]", "", regex=True)
    data = data[data['title'].str.len() > 1]
    for idx, title in enumerate(data.title):
        if not title.isascii():
            data = data.where(data['title'] != title)
            print(title)
    data = data.dropna()
    data.to_csv("../data/data_genres_processed.csv", index=False)