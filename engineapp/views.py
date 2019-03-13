# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:15:47 2019

@author: Nicolas
"""

import os
import pickle

from os import path as os_path
from flask import Flask, render_template, redirect, request, url_for



AUTO_PATH = os_path.abspath(os_path.split(__file__)[0])
LOCAL_PATH =  'C:/users/Nicolas/37.5/perso - Documents/Formation Data Scientist/Parcours Data Scientist/P6 - Texte/P6_text'
PATH = LOCAL_PATH


app = Flask(__name__)
app.config["DEBUG"] = True
app.config.from_object('config')

CT_DIR = app.config['BASESAVE']
print(CT_DIR)
def load_obj(name):
    with open(os.path.join(CT_DIR, name), 'rb') as f:
        mon_depickler = pickle.Unpickler(f)
        return mon_depickler.load()

