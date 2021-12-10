import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')



@st.cache
def prepare_data():     
  #reading csv files
  books = pd.read_csv('Books.csv',encoding = 'Latin-1', error_bad_lines=False)
  users = pd.read_csv('Users.csv',encoding = 'Latin-1', error_bad_lines=False)
  ratings = pd.read_csv('Ratings.csv',encoding = 'Latin-1', error_bad_lines=False)
  #Renaming the columns to easy calling
  books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'],inplace = True,axis = 1)
  books.rename(columns = {'Book-Title':'Title','Book-Author':'Author','Year-Of-Publication':'Year'},inplace = True)
  ratings.rename(columns = {'User-ID':'Id','Book-Rating':'Rating'},inplace = True)
  users.rename(columns = {'User-ID':'Id'},inplace = True)

  #filtering the books which have more than 200 ratings
  x = ratings['Id'].value_counts() > 200
  y = x[x].index
  ratings = ratings[ratings['Id'].isin(y)]

  #Merge ratings with books
  rating_with_books = ratings.merge(books,on = 'ISBN')
  

  #Groupby using title with count of rating column
  number_rating = rating_with_books.groupby('Title')['Rating'].count().reset_index()
  number_rating.rename(columns = {'Rating':'Number_Of_Ratings'},inplace = True)

  #rating merged with rating with books
  final_rating = rating_with_books.merge(number_rating,on = 'Title')
  final_rating = final_rating[final_rating['Number_Of_Ratings'] >= 50]

  #Dropping duplicates
  final_rating.drop_duplicates(['Id','Title'], inplace=True)

  #pivot table 
  book_pivot = final_rating.pivot(columns = 'Id',index = 'Title',values = 'Rating')
  book_pivot.fillna(0,inplace = True)

  #converting pivot table to sparse matrix (storage friendly)
  book_sparse = csr_matrix(book_pivot)

  #Create the model
  model = NearestNeighbors(algorithm='brute')  #brute force algorithm

  #fitting the model
  model.fit(book_sparse)
  return model,book_pivot,books,users,ratings


      

model,book_pivot,book_csv,user_csv,rating_csv = prepare_data()
max_ratings = list(book_pivot.loc[:,:].max())
book_list = list(book_pivot.index)
book_list.insert(0,' ')
st.title('Recommended Reader')
st.markdown('''A smart book recommendation system, made using something special 😉''')
st.image('books.jpg')

recommendations = []
st.sidebar.title('Hey, book-worms! 🤗')
st.sidebar.image('sideimage.jpg')
st.sidebar.markdown('---------------------------------------')
nav = st.sidebar.selectbox('Go To',['Home','Recommender','About'])



if nav == 'Recommender':
  book = st.selectbox("Book that you've read",book_list)
  if book != ' ':
    distances, suggestions = model.kneighbors(book_pivot.loc[book, :].values.reshape(1,-1))
    for i in range(len(suggestions)):
      recommendations.append(book_pivot.index[suggestions[i]])
    with st.spinner('Wait for it...'):
      time.sleep(2)
      
    st.success(book)  #booknames
    st.markdown('##### I know you will praise my choice! 😎')
    recommendations = list(recommendations[0])
    st.markdown('## You should also read')
    for j in range(len(recommendations)):
      st.warning(recommendations[j])

  st.markdown('--------------------------------------------------')
  st.subheader('Developed by Anshika Singh')

    

elif nav == 'Home':
  head = st.container()
  body = st.container()
  bottom = st.container()
  with head:
    st.header('Home')
    st.markdown('--------------------------------------------')
    st.markdown('##### Displaying some bookworm statistics')
  with body:
    col1,col2,col3 = st.columns(3)
    col1.info('Book dataframe')
    col1.write(book_csv.iloc[:20,:])
    col2.error('Users dataframe')
    col2.write(user_csv.iloc[:20,:])
    col3.warning('Ratings dataframe')
    col3.write(rating_csv.iloc[:20,:])

    

    
  

    st.markdown('---------------------------------------------')
    st.subheader('Developed by Anshika Singh')

    
    
    
  

elif nav == 'About':
  st.subheader('About')
  st.markdown('----------------------------------')
  st.markdown('#### Hello friends, I am Anshika Singh🌸.  Are you an avid reader like me and you need some really good recommendations about books? You have come to right place. At Recommended Reader, we will handpick the best books for you that you will definitely enjoy reading!')
  
  st.markdown('--------------------------')
    
  st.subheader('Developed by Anshika Singh')
  

      




