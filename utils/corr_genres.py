from typing import List, Any
import pandas as pd


def getListCorrGenres(correlation):
    genre_list: List[List[str]] = []
    for genre in correlation:
        genre_list.append(correlation[genre].sort_values(ascending=False).iloc[:5].index.to_list())
    return genre_list


if __name__ == "__main__":
    data = pd.read_csv("../data/data_genres.csv")
    # drop every book which does not have a category (books with a category which is not in the top 400)
    data = data.drop(data[data.iloc[:, 3::].sum(axis=1) == 0].index)
    corr = data.iloc[:, 3::].corr()
    corr_list = getListCorrGenres(corr)
    # ['Fantasy', 'Paranormal', 'Magic', 'Supernatural', 'Adventure']
    # ['Adult', 'Romance', 'Teen', 'Contemporary', 'Grade']
    # ['Literature', 'Classics', 'Century', 'Novels', 'American']
    # ['Contemporary', 'Lit', 'Romance', 'Adult', 'Erotica']
    # ['Mystery', 'Thriller', 'Crime', 'Suspense', 'Fiction']
    # ['Novels', 'Comics', 'Literature', 'Fiction', 'Classics']
    # ['Thriller', 'Mystery', 'Crime', 'Suspense', 'Horror']
    # ['Nonfiction', 'Biography', 'History', 'Memoir', 'Science']
    # ['Audiobook', 'Crime', 'Thriller', 'Mystery', 'Suspense']
    # ['Adventure', 'Grade', 'Fantasy', 'Childrens', 'Action']
    # ['Paranormal', 'Supernatural', 'Vampires', 'Fantasy', 'Magic']
    # ['Classics', 'Literature', 'Century', 'Novels', 'American']
    # ['Historical', 'War', 'Literature', 'History', 'Fiction']
    # ['History', 'Nonfiction', 'Biography', 'Politics', 'Memoir']
    # ['Childrens', 'Grade', 'Juvenile', 'Books', 'Animals']
    # ['Suspense', 'Thriller', 'Mystery', 'Crime', 'Action']
    # ['Magic', 'Fantasy', 'Paranormal', 'Supernatural', 'Adventure']
    # ['Humor', 'Comedy', 'Lit', 'Books', 'Childrens']
    # ['Crime', 'Thriller', 'Mystery', 'Suspense', 'Audiobook']
    # ['Memoir', 'Biography', 'Autobiography', 'Nonfiction', 'History']
    # ['Grade', 'Childrens', 'Juvenile', 'Adventure', 'Adult']
    # ['Lit', 'Contemporary', 'Romance', 'Adult', 'Drama']
    # ['Supernatural', 'Paranormal', 'Vampires', 'Fantasy', 'Horror']
    # ['Century', 'Literature', 'Classics', 'Novels', 'American']
    # ['Biography', 'Memoir', 'Autobiography', 'Nonfiction', 'History']
    # ['Teen', 'Adult', 'School', 'Contemporary', 'Romance']
    # ['Horror', 'Supernatural', 'Thriller', 'Paranormal', 'Fantasy']
    # ['Comics', 'Novels', 'Fantasy', 'Comedy', 'Humor']
    # ['School', 'Teen', 'Classics', 'Contemporary', 'Drama']
    # ['American', 'Literature', 'Classics', 'Century', 'Novels']
    # ['Philosophy', 'Nonfiction', 'Psychology', 'Spirituality', 'Science']
    # ['Stories', 'Classics', 'Literature', 'American', 'Horror']
    # ['War', 'Historical', 'History', 'Politics', 'Biography']
    # ['Erotica', 'Romance', 'Contemporary', 'Adult', 'Lit']
    # ['Drama', 'Contemporary', 'Lit', 'School', 'Suspense']
    # ['Vampires', 'Paranormal', 'Supernatural', 'Fantasy', 'Romance']
    # ['Religion', 'Spirituality', 'Christian', 'Philosophy', 'Nonfiction']
    # ['Christian', 'Religion', 'Spirituality', 'Nonfiction', 'Help']
    # ['Science', 'Nonfiction', 'Philosophy', 'History', 'Psychology']
    # ['Juvenile', 'Grade', 'Childrens', 'Books', 'Animals']
    # ['Dystopia', 'Fantasy', 'Teen', 'Fiction', 'Adult']
    # ['Family', 'Contemporary', 'Childrens', 'Juvenile', 'Grade']
    # ['Politics', 'History', 'Nonfiction', 'Philosophy', 'Science']
    # ['Psychology', 'Help', 'Philosophy', 'Nonfiction', 'Science']
    # ['LGBT', 'Contemporary', 'Romance', 'Erotica', 'Poetry']
    # ['Travel', 'Memoir', 'Biography', 'History', 'Adventure']
    # ['Books', 'Childrens', 'Juvenile', 'Animals', 'Grade']
    # ['Action', 'Adventure', 'Thriller', 'Suspense', 'Mystery']
    # ['Spirituality', 'Religion', 'Philosophy', 'Nonfiction', 'Help']
    # ['Autobiography', 'Memoir', 'Biography', 'Nonfiction', 'History']
    # ['Mythology', 'Fantasy', 'Magic', 'Paranormal', 'Supernatural']
    # ['Animals', 'Childrens', 'Books', 'Juvenile', 'Grade']
    # ['Poetry', 'Classics', 'Literature', 'Century', 'American']
    # ['Help', 'Psychology', 'Nonfiction', 'Spirituality', 'Philosophy']
    # ['Comedy', 'Humor', 'Comics', 'Novels', 'Autobiography']



