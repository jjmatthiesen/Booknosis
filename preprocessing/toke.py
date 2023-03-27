import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim.utils import tokenize
import torchtext
from torchtext.data import get_tokenizer
import nltk
import numpy as np
# nltk.download('punkt')


def tokenize_data(df):
    df = df.iloc[:, :2]
    cv = CountVectorizer(stop_words='english', tokenizer=nltk.word_tokenize)
    cv_matrix = cv.fit_transform(df['title'])
    df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['title'].values, columns=cv.get_feature_names_out())
    rating = df.iloc[:, -1]
    df_dtm["rating_value"] = np.array(rating)
    return df_dtm


if __name__ == "__main__":
    data = pd.read_csv("../data/data_genres_processed.csv")
    data_tokenized = tokenize_data(data)

    # cv = CountVectorizer(stop_words='english', tokenizer=nltk.word_tokenize)
    # cv_matrix = cv.fit_transform(data['title'])
    # df_dtm = pd.DataFrame(cv_matrix.toarray(), index=data['title'].values, columns=cv.get_feature_names_out())
    print(data_tokenized.head())

