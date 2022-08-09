
from flask import Flask, jsonify, request
import json

import pandas as pd
import numpy as np
import dill
import ast


app = Flask(__name__)

# Load pre-trained models :
    
def load_preprocessor():
    with open('preprocessing_titles.pkl', 'rb') as file:
        cleaning_title_preprocessor = dill.load(file)
    return(cleaning_title_preprocessor)
    
def load_NMF():
    with open('NMF_model.pkl', 'rb') as file:
        NMF = dill.load(file)
    return(NMF)

def load_tfidf():
    with open('tfidf.pkl', 'rb') as file:
        tfidf = dill.load(file)
    return(tfidf)

def load_tag_topic_matrix():
    Tags_Topics_Matrix = pd.read_csv("Tags_Topics_Matrix.csv")
    Tags_Topics_Matrix["Topic_tags"] = Tags_Topics_Matrix["Topic_tags"].apply(lambda x: ast.literal_eval(x))
    return(Tags_Topics_Matrix)

def load_template():
    df_template = pd.read_csv("df_template.csv")
    return(df_template)
    


@app.route('/predict', methods=['GET'])
def predict():
    
    
    # parse input features from request
    request_json = request.get_json()
    question = str(request_json["input"])
    
    # transform question as a dataframe
    df_question = load_template()
    df_question["Title"][0] = question
    
    # Loading models :
    preprocessor = load_preprocessor()
    NMF = load_NMF()
    Matrix = load_tag_topic_matrix()
    TFIDF = load_tfidf()
    
    # Definition of useful functions :    
    def topics_to_tags(topic_probas, matrix):
        proposed_tags = []
        for detected_topics in topic_probas:
            proposed_tags.append(matrix[matrix['Topic_names']=='Topic '\
                            +str(detected_topics[0])]['Topic_tags'].values[0][:int(detected_topics[1]*10)])
        return(proposed_tags)
    
    def which_topics(text, model):
    
        # Creating probabilities :
        results = model.transform(text)
        
        # Defining the most likely topics with their respective probabilities :
        proba_sum = 0
        detected_topics = []
        for i in range(results.shape[1]):
            round_proba = np.round(results[0][i], 1)
            if round_proba >= 0.1:
                proba = round_proba
                topic_index = i
                proba_sum = proba_sum + proba
                detected_topics.append([topic_index, proba])
                
        # Case where every topic has less than 10% probabilities : choosing the topic with maximum probability :
        if len(detected_topics)==0:
            maxi_h = 0
            maxi_proba = 0
            for h in range(results.shape[1]):
                if results[0][h] > maxi_proba:
                    maxi_h = h
                    maxi_proba = results[0][h]
            detected_topics.append([maxi_h, 1])  
            proba_sum = 1 # to skip next step
                
        # Adjusting the proba associated with the most likely topic so that the sum always equals to 1 :
        if proba_sum < 1: 
            maxi = 0
            maxi_j = 0
            for j in range(len(detected_topics)):
                if detected_topics[j][1] > maxi:
                    maxi = detected_topics[j][1]
                    maxi_j = j
            detected_topics[maxi_j][1] = np.round(detected_topics[maxi_j][1] + (1 - proba_sum), 1)
            
                
        return(detected_topics)
    
    # Preprocessing :
        
    df_question = preprocessor.fit_transform(df_question)
    
    question = df_question["Title"][0]
    
    
    # Giving tags :
    
    tags = topics_to_tags(which_topics(TFIDF.transform(question), NMF), Matrix)[0]
    
    
    response = json.dumps({'response': tags})
    
    return response, 200

if __name__ == '__main__':
    app.run(debug=True)