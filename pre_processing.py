#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Rogerio Shieh Barbosa

"""

import numpy as np 
import os
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

class PreProcess:

    def __init__(self):

        self.stop_words = self.get_stopwords()
        self.punctuations = string.punctuation
        self.parser = English()

    def get_stopwords(self):

        nlp = spacy.load('en')
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        stop_words.update(('subject', 'am', 'pm')) 
        
        return stop_words

    def create_data(self, folder):
    
        data = []
    
        for file in os.listdir(folder + 'ham/'):
            if file[-3::] == 'txt':
                data.append((file, 0, self.corpus_reader(folder + 'ham/' + file))) 

        for file in os.listdir(folder + 'spam/'):
            if file[-3::] == 'txt':
                data.append((file, 1, self.corpus_reader(folder + 'spam/' + file))) 
                
        return np.array(data)

    def corpus_reader(self, corpusfile): 

        with open(corpusfile,'r', encoding='latin-1') as f: 
            read_data = f.read()
            return [s for s in self.spacy_tokenizer(read_data)]

    # Seen in https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
    def spacy_tokenizer(self, sentence): # List(str)) -> List(str):
       
        mytokens = self.parser(sentence)
        
        # Lemmatizing each token and converting each token into lowercase
        mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

        # Removing stop words
        mytokens = [ word for word in mytokens if word not in self.stop_words and word not in self.punctuations and not word.isdigit() ]

        return mytokens

    def pre_process(self, folder):

        return self.create_data(folder)