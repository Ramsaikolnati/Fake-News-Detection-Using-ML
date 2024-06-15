import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def load_data():
    # Load your dataset (fake.csv and true.csv)
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    #inserting a column "class" as target feature
    df_fake["class"] = 0
    df_true["class"] = 1
    # Removing last 10 rows for manual testing
    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480,23470,-1):
           df_fake.drop([i], axis = 0, inplace = True)
    
    
    df_true_manual_testing = df_true.tail(10)
    for i in range(21416,21406,-1):
         df_true.drop([i], axis = 0, inplace = True)
    df_fake_manual_testing["class"] = 0
    df_true_manual_testing["class"] = 1
    df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
    df_manual_testing.to_csv("manual_testing.csv")
    df_merge = pd.concat([df_fake, df_true], axis =0 )
    # Removing columns Which are not required
    df = df_merge.drop(["title", "subject","date"], axis = 1)
    # Random shuffling the DataFrame
    df = df.sample(frac = 1)
    df.reset_index(inplace = True)
    df.drop(["index"], axis = 1, inplace = True)
    df["text"] = df["text"].apply(wordopt)
    # Defining dependent and independent variables
    x = df["text"]
    y = df["class"]
    # Splitting Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)
    LR = LogisticRegression()
    LR.fit(xv_train,y_train)
    return x_train, x_test, y_train, y_test, vectorization, LR, DT, GBC, RFC

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
    pass

def make_prediction(news):
    # Your make_prediction function code
    df["text"] = df["text"].apply(wordopt)
    # Defining dependent and independent variables
    x = df["text"]
    y = df["class"]
    pass
