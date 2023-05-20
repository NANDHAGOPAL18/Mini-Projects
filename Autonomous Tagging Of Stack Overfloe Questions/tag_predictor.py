import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv("datasets/final-data.csv")

df1 = df.dropna()
import ast
df1['Tags'] = df1['Tags'].apply(lambda x:ast.literal_eval(x))

#load from file
tagPredictorModel = joblib.load('data preprocessing/tagPredictorr.pkl')

tfidf = TfidfVectorizer(analyzer = 'word' , max_features=1000, ngram_range=(1,3) , stop_words='english')
X = tfidf.fit_transform(df1['Body'].values.astype(str))

multilabel = MultiLabelBinarizer()
y = df1['Tags']
y = multilabel.fit_transform(y)

def getTags(question):
    question = tfidf.transform(question)
    tags = multilabel.inverse_transform(tagPredictorModel.predict(question))
    print(tags)
    return tags