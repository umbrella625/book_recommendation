
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import streamlit as st

df_books = pd.read_csv(
    '/Books.csv',
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    '/Ratings.csv',
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

print(df_books.shape)
df_books.head()

print(df_ratings.shape)
df_ratings.head()

df_books.dropna(inplace=True)

ratings = df_ratings['user'].value_counts()
df_ratings_rm = df_ratings[~df_ratings['user'].isin(ratings[ratings < 200].index)]

ratings = df_ratings['isbn'].value_counts()
len(ratings[ratings < 100])

books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
         "I'll Be Seeing You",
         "The Weight of Water",
         "The Surgeon",
         "I Know This Much Is True"]

for book in books:
    count = df_ratings_rm['isbn'].isin(df_books[df_books['title'] == book]['isbn']).sum()
    print(f"Occurrences of '{book}': {count}")

df = df_ratings_rm.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T

df.index = df.join(df_books.set_index('isbn'))['title']

df = df.sort_index()
df.loc["The Queen of the Damned (Vampire Chronicles (Paperback))"][:5]

model = NearestNeighbors(metric='cosine')
model.fit(df.values)

title = 'The Queen of the Damned (Vampire Chronicles (Paperback))'

book_features = df.loc[title].values.reshape(1, -1)
book_features = book_features[:, :df.shape[1]]
distance, indice = model.kneighbors(book_features, n_neighbors=6)
pd.DataFrame({
    'title'   : df.iloc[indice[0]].index.values,
    'distance': distance[0]
}) \
.sort_values(by='distance', ascending=False)

def get_recommends(title = ""):
  try:
    book = df.loc[title]
  except KeyError as e:
    print('The given book', e, 'does not exist')
    return

  distance, indice = model.kneighbors([book.values], n_neighbors=6)

  recommended_books = pd.DataFrame({
      'title'   : df.iloc[indice[0]].index.values,
      'distance': distance[0]
    }) \
    .sort_values(by='distance', ascending=False) \
    .head(5).values

  return [title, recommended_books]

def main():
    st.title("Book Recommendation System")

    book_title = st.text_input("Enter the title of the book:")

    if book_title:
        books = get_recommends(book_title)
        if books:
            st.subheader("Recommended Books:")
            for book in books:
                st.write(book)
        else:
            st.write("No recommendations found for the provided book title.")

if __name__ == "__main__":
    main()
