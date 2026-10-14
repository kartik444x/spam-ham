import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

import os
import string
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
df=pd.read_csv("spam.csv",encoding='latin-1')
df=df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})

df['is_spam']=df['label'].map({'ham':0,'spam':1})
print(f'rows={df.shape[0]},columns{df.shape[1]}')
print(df['label'].value_counts())
print('\n first 5 rows')
print(df.head())
def preprocess_text(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in string.punctuation]

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)


df['clean_text'] = df['text'].apply(preprocess_text)

print("Processing complete")

print('\nOriginal:', df['text'].iloc[1])
print('\nCleaned:', df['clean_text'].iloc[1])

x_train,x_test,y_train,y_test=train_test_split(df['clean_text'],df['is_spam'],test_size=0.20,random_state=42,stratify=df['is_spam'])

tfidf=TfidfVectorizer(max_features=2300)
x_train_tfidf=tfidf.fit_transform(x_train)
x_test_tfidf=tfidf.fit_transform(x_test)

nb_classfier=MultinomialNB()
nb_classfier.fit(x_train_tfidf,y_train)
print("naive bayes model trained succesfully")

y_pred=nb_classfier.predict(x_test_tfidf)
accuracy=accuracy_score(y_test,y_pred)
print(f'model test accurcy{accuracy*100:.2f}')

cm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['ham','spam'])
disp.plot(cmap="Blues")
plt.grid(False)
plt.title("spam detection confusion matrix")
plt.show()
custom_text={"hey are we still ,meeting for coffee","hey why are uh ignoring me","this thing dont know how to walk","sojao ajj ke liye itna kaafi hai"}

cleaned=[preprocess_text(msg)for msg in custom_text]
custom_tfidf=tfidf.transform(cleaned)
predictions=nb_classfier.predict(custom_tfidf)

print("custom prediction")
print("-"*80)
for msg,pred in zip(custom_text,predictions):
    result="spam"if pred==1 else "ham"
    print(f"message{msg}")
    print(f'flag:{result}')
    print("-"*80)

tfidf_2gram=TfidfVectorizer( max_features=2500,ngram_range=(1,2))
x_train_2g=tfidf_2gram.fit_transform(x_train)
x_test_2g=tfidf_2gram.transform(x_test)

nb_2g=MultinomialNB().fit(x_train_2g,y_train)
acc_2g=accuracy_score(y_test,nb_2g.predict(x_test_2g))
print(f'1-gram vocabulary accuracy:{accuracy*100:.2f}')
print(f'1 & 2g -gram vocabulary accuracy:{acc_2g*100:.2f}')
print('\n bigrams helps capture context eg good or not good')
from nltk.stem import WordNetLemmatizer
def preprocess_with_lemmatizer(text):
    text = text.lower()

    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # Stop words
    stop_words_list = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words_list]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)


print("Preprocessing dataset using WordNet Lemmatizer")

df['clean_lem'] = df['text'].apply(preprocess_with_lemmatizer)


# Train-test split
x_train_lem, x_test_lem, y_train_lem, y_test_lem = train_test_split(
    df['clean_lem'],
    df['is_spam'],
    test_size=0.2,
    random_state=42,
    stratify=df['is_spam']
)


# TF-IDF
tfidf_lem = TfidfVectorizer(
    max_features=2500,
    ngram_range=(1, 2)
)

x_train_lem_tfidf = tfidf_lem.fit_transform(x_train_lem)
x_test_lem_tfidf = tfidf_lem.transform(x_test_lem)


# Naive Bayes model
nb_lem = MultinomialNB()

nb_lem.fit(x_train_lem_tfidf, y_train_lem)


# Prediction
y_pred_lem = nb_lem.predict(x_test_lem_tfidf)


# Accuracy
acc_lem = accuracy_score(y_test_lem, y_pred_lem)


print(f"\nStemming pipeline accuracy: {accuracy* 100:.2f}%")
print(f"Lemmatization pipeline accuracy: {acc_lem * 100:.2f}%")

lr_model = LogisticRegression(random_state=42)

# Train the model
lr_model.fit(x_train_tfidf, y_train)

# Prediction
lr_pred = lr_model.predict(x_test_tfidf)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)

# Results
print(f"Naive Bayes accuracy: {accuracy * 100:.2f}%")
print(f"Logistic Regression test accuracy: {lr_accuracy * 100:.2f}%")

print("\nClassification Report for Logistic Regression")
print(classification_report(
    y_test,
    lr_pred,
    target_names=['ham', 'spam']
))