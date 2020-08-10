import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def tf_idf(dataframe, label):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe.loc[:, label])
    with open('tf_idf','wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    with open('tf_idf_vec','wb') as f:
        pickle.dump(tfidf_vectorizer, f)

df = pd.read_csv('recipes.csv')
datas = []
for index, row in df.iterrows():
    ings = row['ingredients']
    s=re.sub(r'\(.*oz.\)|kg|½|¾|¼|-|crushed|crumbles|ground|minced|tsp|tbsp|required|powder|chopped|sliced|pinch|cups|cup|/|ml|[0-9]',
                             '', 
                             str(ings))
    
    #Remove Digits
    s=re.sub(r"(\d)", "", s)
    
    #Remove content inside paranthesis
    s=re.sub(r'\([^)]*\)', '', s)
    
    
    s=s.lower()
    
    #Remove Stop Words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(s)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    s= ' '.join(filtered_sentence)
    
    ings = s
    data = {}
    data['recipie'] = row['name']
    data['ingredients'] = ings
    datas.append(data)

df = pd.io.json.json_normalize(datas)


tf_idf(df, 'ingredients')