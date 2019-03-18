# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:15:47 2019

@author: Nicolas
"""

import os
import pickle
import nltk

import numpy as np

from os import path as os_path
from flask import Flask, render_template, redirect, request, url_for
from nltk.stem.snowball import EnglishStemmer
from bs4 import BeautifulSoup


PATH = os_path.abspath(os_path.split(__file__)[0])


app = Flask(__name__)
app.config["DEBUG"] = True
app.config.from_object('config')


CT_DIR = app.config['BASESAVE']
print(CT_DIR)

def load_obj(name):
    with open(os.path.join(CT_DIR, name), 'rb') as f:
        mon_depickler = pickle.Unpickler(f)
        return mon_depickler.load()

#Chargement des pickles
TAG_LIST = load_obj('tags_list')
STOP_WORDS = load_obj('stop_words')
LDA_MODEL = load_obj('lda_model')
TF_LIST = load_obj('TF')
CLF_LIST = load_obj('clf')

#Extraction des données des pickles
TF = TF_LIST[0]
TF_FEATURE_NAMES = TF_LIST[1]
LDA_COMPONENTS = LDA_MODEL.components_ # p(words|topic)
CLF = CLF_LIST[0]
TFIDF = CLF_LIST[1]

#Stemmatisation
stemmer = EnglishStemmer()

def intersect_stem(text,sw):
    
    """Fonction effectuant 2 opérations :
    - la première permettant de supprimer les stopwords de la première liste
    - la seconde permettant de "stemmatiser" les termes de la liste (en langue 
    anglaise)
    text : texte à transformer
    sw : liste des stopwords
    """
    
    final_list = [stemmer.stem(k) for k in text if k not in sw]
    return final_list


def format_question(quest):

        """fonction permettant d'appliquer tous les formatages à la question
        saisie
        """

        #Formatage de la question
        #nettoyage html
        QUESTION_CLEAN =  BeautifulSoup(quest , 'html.parser').get_text()

        #Tokensiation
        tokenizer = nltk.RegexpTokenizer(r'[a-z0-9+#]+')
        QUESTION_TOK = tokenizer.tokenize(QUESTION_CLEAN.lower())

        QUESTION_TOK_LIST = intersect_stem(QUESTION_TOK, STOP_WORDS)

        FINAL_DOC = ' '.join(QUESTION_TOK_LIST)
        
        return FINAL_DOC
        

def format_question_lda(quest):
        
        """Fonction appliquant le formatage nécessaire pour la méthode LDA
        """
        
        LDA_OUTPUT = LDA_MODEL.transform(TF.transform([quest])) #p(topic|document)
        return LDA_OUTPUT[0]

def format_question_clf(quest):
        
        """Fonction appliquant le formatage nécessaire pour la méthode supervisée.
        """

        X = TFIDF.transform([quest])
        return X


def lda_tag_doc(quest, n):
    
    """Fonction permettant de renvoyer les n tags les plus pertinents 
    pour un document donné
    """
    
    lda_tags = []
    step1 = format_question_lda(quest)*LDA_COMPONENTS.T
    step2 = step1.sum(axis=1)
    lda_tags = [TF_FEATURE_NAMES[i] for i in np.argsort(-step2)[:n]]
    lda_tags_str = ', '.join(lda_tags)
    return lda_tags_str


def clf_tag(quest):
    
    """Fonction permettant de renvoyer les tags issus de la méthode supervisée
    """
    
    doc = format_question_clf(quest)
    n = CLF.predict(doc).sum()
    recos = [TAG_LIST[i] for i in np.argsort(-CLF.predict(doc)[0])[:n] ] 
    clf_tags_str = ', '.join(recos)
    return clf_tags_str


def result(a,b):
    
    """Fonction permettant de renvoyer les tags issus de la méthode supervisée
    """

    prompt1 = "Nous avons trouvé les tags existants suivants :\n"
    prompt2 = "\nNous vous recommandons également les mots clés suivants :\n"
    prompt = prompt1 + a + prompt2 + b
    return prompt

def retjson(a):
    #python2json = json.dumps(a)
    return a
    #return python2json

comments = [""]
@app.route('/' , methods=["GET" , "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html" , comments = comments )

    QUESTION = str(request.form["question"])
    
    #reco_tag = lda_tag_doc(QUESTION, 20)
    #reco_tag = clf_tag(QUESTION)
    reco_tag = result(clf_tag(QUESTION), lda_tag_doc(QUESTION, 20))
    comments[0] = reco_tag 
    return retjson(reco_tag)


if __name__ == "__main__":
    app.run()