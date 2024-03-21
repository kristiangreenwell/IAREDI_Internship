"""
Author: Kristian Greenwell
Date: 08/03/2023
IA-REDI Internship Summer 2023
Description: This program uses machine learning to detect whether an email is spam/phishing or not.
"""

# Import needed libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib

''' This section of code creates, trains, saves and test the model for accuracy. '''
train = pd.read_csv("mail_data.csv")

X = train['Message']
y = train['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

for i in range(len(X_train)):
    text = X_train.iloc[i]
    text = text.lower()
    wordlist = text.split()
    text = ' '.join([word for word in wordlist if word not in stop_words])
    X_train.iloc[i] = text

for i in range(len(X_test)):
    text = X_test.iloc[i]
    text = text.lower()
    wordlist = text.split()
    text = ' '.join([word for word in wordlist if word not in stop_words])
    X_test.iloc[i] = text

cv = CountVectorizer()
features = cv.fit_transform(X_train)

LR = LogisticRegression()
LR.fit(features, y_train)

filename = 'spamDetectionModel.sav'
joblib.dump(LR, open(filename, 'wb'))
load_model = joblib.load(open(filename, 'rb'))

features_test = cv.transform(X_test)
pred=load_model.predict(features_test)

print(accuracy_score(y_test, pred))
print(f1_score(y_test, pred, pos_label='spam'))

''' This section implements the code. '''
load_model = joblib.load(open(filename, 'rb'))

email1 = "http//tms. widelive.com/index. wml?id=820554ad0a1705572711&first=true¡C C Ringtone¡"
email2 = "Ok lor."

inbox = pd.DataFrame({'Message': [email1, email2]})

Xnew = inbox['Message']
for i in range(len(Xnew)):
    text = Xnew.iloc[i]
    text = text.lower()
    wordlist = text.split()
    text = ' '.join([word for word in wordlist if word not in stop_words])
    Xnew.iloc[i] = text
features = cv.transform(Xnew)
ynew = load_model.predict(features)
for i in range(len(Xnew)):
    print("Email", i+1, "is", ynew[i])




