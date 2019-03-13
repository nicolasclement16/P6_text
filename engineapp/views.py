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


AUTO_PATH = os_path.abspath(os_path.split(__file__)[0])
LOCAL_PATH =  'C:/users/Nicolas/37.5/perso - Documents/Formation Data Scientist/Parcours Data Scientist/P6 - Texte/P6_text'
PATH = LOCAL_PATH


app = Flask(__name__)
app.config["DEBUG"] = True
app.config.from_object('config')

#Chargment des objets pickle

CT_DIR = app.config['BASESAVE']
print(CT_DIR)
def load_obj(name):
    with open(os.path.join(CT_DIR, name), 'rb') as f:
        mon_depickler = pickle.Unpickler(f)
        return mon_depickler.load()

TAG_LIST = load_obj('tags_list')
STOP_WORDS = load_obj('stop_words')
LDA_MODEL = load_obj('lda_model')
TF_LIST = load_obj('TF')

#Saisie de la question
QUESTION = input("Type your question in English")

#Formatage de la question
#nettoyage html
QUESTION_CLEAN =  BeautifulSoup(QUESTION , 'html.parser').get_text()

#Tokensiation
tokenizer = nltk.RegexpTokenizer(r'[a-z0-9+#]+')
QUESTION_TOK = tokenizer.tokenize(QUESTION_CLEAN.lower())

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

QUESTION_TOK_LIST = intersect_stem(QUESTION_TOK, STOP_WORDS)

FINAL_DOC = ' '.join(QUESTION_TOK_LIST)

#Génération des mots recommandés
TF = TF_LIST[0]
TF_FEATURE_NAMES = TF_LIST[1]

LDA_OUTPUT = LDA_MODEL.transform(TF.transform([FINAL_DOC])) #p(topic|document)
LDA_COMPONENTS = LDA_MODEL.components_ # p(words|topic)

def lda_tag_doc(n):
    
    """Fonction permettant de renvoyer les n tags les plus pertinents 
    pour un document donné
    """
    
    lda_tags = []
    step1 = LDA_OUTPUT[0]*LDA_COMPONENTS.T
    step2 = step1.sum(axis=1)
    lda_tags = [TF_FEATURE_NAMES[i]for i in np.argsort(-step2)[:n]]
    return lda_tags

lda_tag_doc(10)
