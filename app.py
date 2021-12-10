import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from func_file import *
from string import punctuation
from nltk.corpus import stopwords

#####
df = pd.read_csv('prog_book.csv')
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')
stop = set(stop)

df['clean_Book_title']=df['Book_title'].apply(clean_text)
df['clean_Description']=df['Description'].apply(clean_text)

vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
X = vectorizer.fit_transform(df['clean_Book_title'])
title_vectors = X.toarray()

tk = 0
st.title('Book Recommendation System')

col1, col2 = st.columns(2)
#taking book name as input
with col1:
    book = st.text_input('Enter book name that you liked : ')
    
#taking multiple fiels to get similarity
with col2:
    feat = st.selectbox("Select Mode : ",['Book_title', 'Rating', 'Price'])
    if st.button('Search'):
        tk = 1

#st.dataframe(df.head(10))
if tk == 1:
    st.success('Recommending books similar to '+book)
    rec = st.empty()
    rec = st.dataframe(get_recommendations(book, 'Book_title', df, title_vectors, feat), width=700, height=76)

#print(get_recommendations())