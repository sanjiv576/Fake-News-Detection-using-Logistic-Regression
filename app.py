import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords # the for of in with
from nltk.stem.porter import PorterStemmer # loved loving == love
from sklearn.feature_extraction.text import TfidfVectorizer # loved = [0.0]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# read csv file
news = pd.read_csv("/Users/uSer/Desktop/ai/fake news/WELFake_Dataset.csv")
# remove unnecessary attributes
news = news.drop(labels=["Unnamed: 0", "title"], axis=1)
# rename label 
news['label'] = news['label'].map({1: 'Real', 0: 'Fake'})
# remove null values
news = news.dropna()

# stemming function

ps = PorterStemmer()

# define a function for stemming
def stemming(text):
    
#     store only lower case a-z and upper case A-Z, ignoring other words/letters, finally convert into lowercase
    stemmedContent = re.sub('[^a-zA-Z]', ' ', text).lower()
#     split the word
    stemmedContent = stemmedContent.split()
    
#     check whether each stemmed data is stopword or not, if yes remove it and if no store it
    stemmedContent = [ps.stem(word) for word in stemmedContent if not word in stopwords.words('english')]
#     join stemmed data by a space
    stemmedContent = ' '.join(stemmedContent)
    return stemmedContent


# apply stemming function to the text attribute
news['text'] = news['text'].apply(stemming)

# For Vectorization
# separate the data and label
X = news['text'].values
y = news['label'].values

# vectorization  ==> converting textual data into numerical data
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Separation of training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 2)

# Training a model using Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train,y_train)


# create a website

st.title('Fake News Detection System')

inputText = st.text_input('Enter news ')

def prediction(inputText):
    data = vector.transform([inputText])
    prediction = model.predict(data)
    return prediction[0]

if inputText:
    # calls above prediction function
    result = prediction(inputText)
    if result == 1:
        st.write('The news is Fake')
    else:
        st.write('The news is Real')