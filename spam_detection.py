import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#Data import
ds = pd.read_csv('spam.csv')
print(ds)
print(ds.isnull().sum())
ds['spam'] = ds['Category'].apply(lambda x:1 if x == 'spam' else 0)
print(ds)

#Splitting data into indpendent and dependent features to train model
X = ds.Message
y = ds.spam
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#pipelines
pipe = Pipeline([
    ('v', CountVectorizer()),
    ('nb', MultinomialNB())
])

#training model
pipe.fit(X_train, y_train)

#input
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]

#predicting the input
y_preds = pipe.predict(emails)

#chehcking the score of the model
print(pipe.score(X_test, y_test))

#output if 1: spam, 0: not spam 
print(y_preds)