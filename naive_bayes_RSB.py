#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Rogerio Shieh Barbosa

Many thanks to Aisha Javed (https://towardsdatascience.com/na%C3%AFve-bayes-from-scratch-using-python-only-no-fancy-frameworks-a1904b37222d),
where I got an idea about how to organize the class and functions.
"""

import pandas as pd 
import numpy as np 
import time
import typing
import os
from collections import defaultdict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
from sklearn.model_selection import train_test_split

class NaiveBayes:
    
    def __init__(self,unique_classes):
        
        self.classes = unique_classes #  unique number of classes on the training set
        
    def addToBow(self,example,dict_index):
        
        '''Adds every tokenized word to dictionary/BoW corresponding to its label  

        Parameters
        ----------
        example: list(str) 
            pre-processed tokens in an email            
        dict_index: int 
            represents label (0: ham, 1: spam)
        
       '''
     
        for token_word in example:
          
            self.bow_dict[dict_index][token_word] += 1 

    def train(self,dataset):
        
        '''
        Training function which will compute a BoW for each category/class (train the Naive Bayes Model) 
       
        Parameters
        ----------
        dataset: list(tuple(str, list(str) ))
            list with tuple (filename, list of pre-processed tokens)

        '''
    
        self.examples = dataset
        
        self.bow_dict = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        
        #label: (doc_count, word_count)
        self.count_dict = {'ham': [0, 0], 'spam': [0,0]}

        for i in range(len(self.examples)):
        
            ham_examples = []
            spam_examples = []
            
            if self.examples[i][1] == 0: #ham
                ham_examples.append(self.examples[i])
                self.addToBow(self.examples[i][2], 'ham')
                self.count_dict['ham'][0] += 1
                self.count_dict['ham'][1] = len(self.examples[i][2])
                
             
            elif self.examples[i][1] == 1:
                spam_examples.append(self.examples[i])
                self.addToBow(self.examples[i][2], 'spam')
                self.count_dict['spam'][0] += 1
                self.count_dict['spam'][1] = len(self.examples[i][2])
        

        total_emails = self.count_dict['ham'][0] + self.count_dict['spam'][0]
        self.prior_ham = self.count_dict['ham'][0] / total_emails
        self.prior_spam = self.count_dict['spam'][0] / total_emails
                
        
    def getExampleProb(self,test_example): #list(str)) -> int:                               
        
        '''
        Estimates posterior probability of the given test example.

        Returns:
        ---------
        0 if prediction is ham
        1 if prediction is spam
        '''                                      
                                              
        log_word_prob_spam = 0.
        log_word_prob_ham = 0.
        
        for word in test_example:
            prob_word_ham = np.log ( (self.bow_dict['ham'][word] + 1) / (self.count_dict['ham'][1] + 1) )
            prob_word_spam = np.log ( (self.bow_dict['spam'][word] + 1) / (self.count_dict['spam'][1] + 1) )
            
            log_word_prob_spam += prob_word_spam
            log_word_prob_ham += prob_word_ham
        
        
        log_word_prob_ham *= self.prior_ham
        
        log_word_prob_spam *= self.prior_spam
        
        prob_list = [log_word_prob_ham, log_word_prob_spam]
        
        return prob_list.index(max(prob_list))
    
    def test(self,test_set):     
     
        res = {"correct_spam": 0, "correct_ham": 0, "wrong_spam": 0, "wrong_ham": 0}
        
        for example in test_set:
            
            if example[0].split('.')[-2] == 'ham':
                label = 0
            else:
                label = 1
            
        
            # get the posterior probability of every example                                  
            post_prob = self.getExampleProb(example[2]) #get prob of this example for both classes
            
            if label == post_prob: #correct prediction
                if label == 0:
                    res["correct_ham"] += 1 #true neg
                else:
                    res["correct_spam"] += 1 #true pos
            else:
                if label == 0:
                    res["wrong_ham"] += 1 #false neg
                else:
                    res["wrong_spam"] += 1 #false pos

        total = len(test_set)
        correct = res["correct_spam"] + res["correct_ham"]
        accuracy = correct / total
        precision = res["correct_spam"] / (res["correct_spam"] + res["wrong_spam"])
        recall = res["correct_spam"] / (res["correct_spam"] + res["wrong_ham"])
        f1_score = (2 * precision * recall) / (precision + recall)

        print("Total number of emails in test set = {}".format(total))
        print("Total number of correct predictions = {}".format(correct))
        print("Accuracy = {}".format(accuracy))
        print("Precision = {}".format(precision))
        print("Recall = {}".format(recall))
        print("F1 Score = {}\n".format(f1_score))

        return [total, correct, accuracy, precision, recall, f1_score]

if __name__ == '__main__':
    
    start_time = time.process_time()
    
    print("PRE-PROCESSING DATA...")
    print("-------------------------------------")
    pre_process = PreProcess()
    folder = 'data/'
    data = pre_process.pre_process(folder)

    splits = [.9, .8, .7, .6]
    results = []

    for split in splits:
        train, test = train_test_split(data, test_size = split)
        print("-------------------------------------")
        print('Training/Testing using Naive Bayes, split = {}/{}'.format(split, (1-split) ))
        print("-------------------------------------\n")
        nb = NaiveBayes(np.unique([0,1]))
        nb.train(train)
        res = nb.test(test) #[total, correct, accuracy, precision, recall, f1_score]
        results.append(res)
        print("\n")

    end_time = time.process_time()
    print("\nProgram completed in {} seconds.".format(end_time - start_time))

    

    
    