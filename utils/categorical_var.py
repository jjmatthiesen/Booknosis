import pandas as pd
import time
import calendar


def getOrdinal(df, column_name, file_name="", save=False):
    """
    :param save: should file be saved?
    :param df: dataframe
    :param column_name: name of the column which should be transformed
    :param file_name: file name for saving the transformed data
    """
    low_quan = df[column_name].quantile(q=0.333)
    height_quan = df[column_name].quantile(q=0.666)
    column_name_ordinal = column_name + "_ordinal"
    df[column_name_ordinal] = 1
    df[column_name_ordinal][df[column_name] < low_quan] = 0
    df[column_name_ordinal][df[column_name] > height_quan] = 2
    df_ordinal = df.drop(columns=[column_name])
    if save:
        if file_name != "":
            df_ordinal.to_csv("../data/data_" + file_name + "_ordinal.csv", index=False)
        else:
            df_ordinal.to_csv("../data/data_" + str(calendar.timegm(time.gmtime())) + "_ordinal.csv", index=False)
    return df_ordinal


if __name__ == "__main__":
    f_name = "data_genres"
    data = pd.read_csv("../data/" + f_name + ".csv")
    data = data.iloc[:, :2]
    # split all ratings over all genres into categorical variables
    data_ordinal = getOrdinal(data, "rating", f_name)