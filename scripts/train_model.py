import pandas as pd
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return ' '.join(ps.stem(i) for i in y)

df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0,1])
df.columns = ['target','text']
df['target'] = df['target'].map({'ham':0,'spam':1})
df = df.drop_duplicates(keep='first')
df['transformed_text'] = df['text'].apply(transform_text)

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text'])
y = df['target'].values
mnb = MultinomialNB()
mnb.fit(X, y)

pickle.dump(tfidf, open('vectorizer.pkl','wb'))
pickle.dump(mnb, open('model.pkl','wb'))
print(f'Model retrained! Vectorizer vocab size: {len(tfidf.vocabulary_)}')
