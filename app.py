from flask import Flask, render_template, request, redirect, url_for
import mysql.connector
import ml_model

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

con=mysql.connector.connect(database='fakenews',user='root',password='')
cur=con.cursor()

app = Flask(__name__)

# Load the machine learning model and data
#x_train, x_test, y_train, y_test, vectorization, LR, DT, GBC, RFC = ml_model.load_data()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Home')
def Home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/AdminLogin')
def admin_login():
    return render_template('AdminLogin.html')

@app.route('/AdminLoginDB',methods=['POST'])
def admin_logindb():
    un=request.form['username']
    pwd=request.form['pwd']
    if un=='admin' and pwd=='12345':
        return render_template('AdminHome.html')
    else:
        return render_template('AdminLogin.html',msg='Pls Check Your Credentials')


@app.route('/UserLogin')
def user_login():
    return render_template('UserLogin.html')

@app.route('/AboutUs')
def about_us():
    return render_template('AboutUs.html')

@app.route('/signup')
def signup():
    return render_template('UserReg.html')

@app.route('/UserRegDB',methods=['POST'])
def UserRegDB():
    name=request.form['name']
    mail=request.form['mail']
    contact=request.form['contact']
    pwd=request.form['pwd']
    s="insert into users(name,email,contact,password) values('"+name+"','"+mail+"','"+contact+"','"+pwd+"')"
    cur.execute(s)
    con.commit()
    return render_template('UserLogin.html')

@app.route('/UserLoginDB',methods=['POST'])
def user_logindb():
    un=request.form['username']
    pwd=request.form['pwd']
    s="select * from users where email='"+un+"' and password='"+pwd+"'"
    cur.execute(s)
    d=cur.fetchall()
    
    if len(d) >0:
        return render_template('UserHome.html')
    else:
        return render_template('UserLogin.html',msg='Pls Check Your Credentials')


@app.route("/uploadDB", methods=['POST'])
def uploadDB():
    f = request.files['fname']
    f1 = request.files['fname1']
    print("name is ",f.filename)
    df = pd.read_csv(f, encoding="utf8")
    df.to_csv(f.filename, index=False)

    df = pd.read_csv(f1, encoding="utf8")
    df.to_csv(f1.filename, index=False)
    return render_template("AdminHome.html")

@app.route("/ViewDataset")
def viewdataset():
    import csv
    lst=[]
    lst1=[]
    with open("./fake.csv", 'r', encoding="utf8") as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        lst.append(row)
    with open("./true.csv", 'r', encoding="utf8") as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        lst1.append(row)
    print("Fake  ",lst)
    print("Fake  ",lst1)
    return render_template("viewdataset.html",data=lst,data1=lst1)

@app.route("/train")
def train():
    global x_train, x_test, y_train, y_test
    global LR,DT,GBC,RFC
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")


    #inserting a column "class" as target feature
    df_fake["class"] = 0
    df_true["class"] = 1



    df_fake.shape, df_true.shape


    # Removing last 10 rows for manual testing
    df_fake_manual_testing = df_fake.tail(10)
    for i in range(23480,23470,-1):
        df_fake.drop([i], axis = 0, inplace = True)
        
        
    df_true_manual_testing = df_true.tail(10)
    for i in range(21416,21406,-1):
        df_true.drop([i], axis = 0, inplace = True)


    df_fake.shape, df_true.shape


    df_fake_manual_testing["class"] = 0
    df_true_manual_testing["class"] = 1


    df_fake_manual_testing.head(10)


    df_true_manual_testing.head(10)


    df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
    df_manual_testing.to_csv("manual_testing.csv")


    #Merging True and Fake Dataframes
    df_merge = pd.concat([df_fake, df_true], axis =0 )
    df_merge.head(10)


    df_merge.columns


    # Removing columns Which are not required
    df = df_merge.drop(["title", "subject","date"], axis = 1)


    df.isnull().sum()


    # Random shuffling the DataFrame
    df = df.sample(frac = 1)



    df.head()



    df.reset_index(inplace = True)
    df.drop(["index"], axis = 1, inplace = True)


    df.columns


    df.head()

    # Creating a function to process the Texts
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

    df["text"] = df["text"].apply(wordopt)


    # Defining dependent and independent variables
    x = df["text"]
    y = df["class"]


    # Splitting Training and Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


    # Convert text to vectors
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    # Logistic Regression
    from sklearn.linear_model import LogisticRegression

    LR = LogisticRegression()
    LR.fit(xv_train,y_train)


    pred_lr=LR.predict(xv_test)

    lracc=accuracy_score(pred_lr, y_test, normalize=False)

    LR.score(xv_test, y_test)


    print(classification_report(y_test, pred_lr))



    # Decision tree classification
    from sklearn.tree import DecisionTreeClassifier

    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)


    pred_dt = DT.predict(xv_test)


    DT.score(xv_test, y_test)


    dtacc=accuracy_score(pred_dt, y_test, normalize=False)
    print(classification_report(y_test, pred_dt))


    # Gradient Boosting classifier
    from sklearn.ensemble import GradientBoostingClassifier
    GBC = GradientBoostingClassifier(random_state=0)
    GBC.fit(xv_train, y_train)


    pred_gbc = GBC.predict(xv_test)



    GBC.score(xv_test, y_test)


    gbcacc=accuracy_score(pred_gbc, y_test, normalize=False)
    print(classification_report(y_test, pred_gbc))


    # Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier

    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(xv_train, y_train)



    pred_rfc = RFC.predict(xv_test)


    RFC.score(xv_test, y_test)

    rfcacc=accuracy_score(pred_rfc, y_test, normalize=False)

    print(classification_report(y_test, pred_rfc))
    accuracy_score(y_true, y_pred, normalize=False)
    return render_template("train.html")
@app.route("/accuracy")
def accuracy():
    #1
    pred_lr=LR.predict(xv_test)
    lracc=accuracy_score(pred_lr, y_test, normalize=False)

    # 2

    pred_dt = DT.predict(xv_test)

    dtacc=accuracy_score(pred_dt, y_test, normalize=False)
    
    # 3
    pred_gbc = GBC.predict(xv_test)


    gbcacc=accuracy_score(pred_gbc, y_test, normalize=False)
    #4
    pred_rfc = RFC.predict(xv_test)


    rfcacc=accuracy_score(pred_rfc, y_test, normalize=False)

    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news_text']
        pred_LR, pred_DT, pred_GBC, pred_RFC = ml_model.make_prediction(news)
        return render_template('result.html', pred_LR=pred_LR,
                               pred_DT=pred_DT,
                               pred_GBC=pred_GBC,
                               pred_RFC=pred_RFC)

    return render_template('predict.html')

@app.route('/logout')
def logout():
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
