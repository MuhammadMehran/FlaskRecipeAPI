import json
import pandas as pd
from pandas.io.json import json_normalize
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, request, jsonify,abort
import os


app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)


# with open('recipes.json', encoding="utf8") as f:
#     data = json.load(f)

# df = json_normalize(data[2]['data'])

df = pd.read_csv('recipes.csv')

datas = []
r_data = {}
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
    
    ings = s.split('~')
    data = {}
    data['recipie'] = row['name']
    data['ingredients'] = ings
    data['img'] = row['image']
    r_data[row['name']] = row['image']
    data['all_ings'] = row['ingredients']
    data['directions'] = row['method']
    data['video'] = row['video']
    datas.append(data)

df = json_normalize(datas)

ings = []
mod = []
for index, row in df.iterrows():
    ings.extend(row['ingredients'])
    mod.append(' '.join(row['ingredients']))

df['ing_mod'] = mod

with open('tf_idf','rb') as f:
    new_tf_idf = pickle.load(f)

with open('tf_idf_vec','rb') as f:
    new_tf_idf_vec = pickle.load(f)


@app.route('/recommend/<path:id>', methods=['GET'])
def get_recommendations(id):
    p_ings= id.lower().split('@')
    datas = []
    for p in p_ings:
        query = new_tf_idf_vec.transform([p])
        cs = cosine_similarity(query, new_tf_idf)
        similarity_list = cs[0]
        result_list = []
        x = similarity_list
        index = np.argwhere(x!=0)
        data = {}
        for ind in index:
            data[ind[0]] = x[ind][0]
        y = {k: v for k, v in sorted(data.items(), key=lambda item: item[1], reverse=True)}
        datas.append(y)

    res = []
    for data in datas:
        for k in data.keys():
            res.append(df.iloc[k]['recipie'])

    new_res = list(set([i for i in res if res.count(i)>1]))
    datas = []

    for r in new_res:
        data = {}
        data['recipe'] = r
        data['img'] = r_data[r]
        datas.append(data)
    return jsonify({'recipes' : datas})
    # return jsonify({'recipes' : []})

@app.route('/info/<path:id>', methods=['GET'])
def get_info(id):
    row = df.loc[df['recipie'] == id].iloc[0]
    print(row)
    data = {}
    data['name'] = id
    data['img'] = row['img']
    data['ings']= row['all_ings']
    data['dir'] = row['directions']
    data['video'] = row['video']
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')