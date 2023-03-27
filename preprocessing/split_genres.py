"""
This file generated several CSVs for every genre each.
"""

import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("../data/data_genres_processed.csv")
    genres_list = data.columns[3::]
    df_fiction = data[data['Fiction'] == 1]
    df_fiction = df_fiction.iloc[:,:2]
    for genre in genres_list:
        exec(f'df_{genre} = data[data[genre] == 1]')
        exec(f'df_{genre} = df_{genre}.iloc[:,:2]')
        exec(f'df_{genre}.to_csv("../data/df_genres/df_" + genre + ".csv", index=False)')

