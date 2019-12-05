#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: rogerio shieh barbosa

"""

import time
import pandas as pd 
import numpy as np 
import os
import string
import heapq
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist


class NearestNeighbors:

    def __init__(self, dataset, k):
        
        self.k = k
        self.bow = {} #key: email filename; values: (label, bow )
        self.vocabulary, self.vocab_idx_dict = self.get_vocabulary(dataset)
        
    def get_vocabulary(self, dataset):
        '''
        Returns: list(str) containing all vocab words
        '''
    
        vocabulary = []
        
        for example in dataset:
            for w in example[2]:
                vocabulary.append(w)  
        
        vocabulary = list(set(vocabulary))
        vocab_idx_dict = {}
        
        for i in range(len(vocabulary)):
            vocab_idx_dict[vocabulary[i]] = i
        
        return vocabulary, vocab_idx_dict
    
    def bag_of_words(self, example):
        '''
        example: list(str) -- 
        vocabulary: list(str) -- contains all vocabulary words
        '''
        
        
        bag = np.zeros(len(self.vocabulary))

        for word in example:
            if word in self.vocab_idx_dict:
                bag[self.vocab_idx_dict[word]] += 1         
            
        return bag
            
    '''
            This was the previous version. Too slow.
         for word in example:
            for i,w in enumerate(self.vocabulary):
                if word == w: 
                    bag[i] += 1
    '''
                   
        
    
    def train (self, dataset):
        
        for example in dataset:
            self.bow[example[0]] = (example[1], self.bag_of_words(example[2]) )
        
    def test_L1 (self, test_example) :
        '''
        test_example: list(str)
        '''
        
        bow_example = self.bag_of_words(test_example)
        
        dict_distances = {}
        
        for email in self.bow.keys():
            dict_distances[email] = np.sum(abs(bow_example - self.bow[email][1]))
        
        k_neighbors = heapq.nsmallest(self.k, dict_distances, key=dict_distances.get)
        
        spam_count = 0
        ham_count = 0
        
        for val in k_neighbors:
            if val.split('.')[-2] == 'ham':
                ham_count += 1
            elif val.split('.')[-2] == 'spam':
                spam_count += 1
 
        return 1 if max(spam_count, ham_count) == spam_count else 0 
        
    def test_L2 (self, test_example) :
        '''
        test_example: list(str)
        '''
        
        bow_example = self.bag_of_words(test_example)
        
        dict_distances = {}
        
        for email in self.bow.keys():
            dict_distances[email] = np.sqrt(np.sum( (bow_example - self.bow[email][1])**2) )
        
        k_neighbors = heapq.nsmallest(self.k, dict_distances, key=dict_distances.get)
        
        spam_count = 0
        ham_count = 0
        
        for val in k_neighbors:
            if val.split('.')[-2] == 'ham':
                ham_count += 1
            elif val.split('.')[-2] == 'spam':
                spam_count += 1
 
        return 1 if max(spam_count, ham_count) == spam_count else 0        
#    
    def test_L_inf (self, test_example) :
        '''
        test_example: list(str)
        '''
        
        bow_example = self.bag_of_words(test_example)
        
        dict_distances = {}
        
        for email in self.bow.keys():
            dict_distances[email] = np.amax(abs( bow_example - self.bow[email][1] ) )
        
        k_neighbors = heapq.nsmallest(self.k, dict_distances, key=dict_distances.get)
        
        spam_count = 0
        ham_count = 0
        
        for val in k_neighbors:
            if val.split('.')[-2] == 'ham':
                ham_count += 1
            elif val.split('.')[-2] == 'spam':
                spam_count += 1
 
        return 1 if max(spam_count, ham_count) == spam_count else 0  
        

    def get_scores(self, dict_res, total):

        correct = dict_res["correct_spam"] + dict_res["correct_ham"]
        accuracy = correct / total
        precision = dict_res["correct_spam"] / (dict_res["correct_spam"] + dict_res["wrong_spam"])
        recall = dict_res["correct_spam"] / (dict_res["correct_spam"] + dict_res["wrong_ham"])
        f1_score = (2 * precision * recall) / (precision + recall)

        print("Total number of emails in test set = {}".format(total))
        print("Total number of correct predictions = {}".format(correct))
        print("Accuracy = {}".format(accuracy))
        print("Precision = {}".format(precision))
        print("Recall = {}".format(recall))
        print("F1 Score = {}\n".format(f1_score))

        return [total, correct, accuracy, precision, recall, f1_score]
    
    def test(self, test_set):
        
        correct_dict = defaultdict(float)
        res_L1 = {"correct_spam": 0, "correct_ham": 0, "wrong_spam": 0, "wrong_ham": 0}
        res_L2 = {"correct_spam": 0, "correct_ham": 0, "wrong_spam": 0, "wrong_ham": 0}
        res_Linf = {"correct_spam": 0, "correct_ham": 0, "wrong_spam": 0, "wrong_ham": 0}
        
        for example in test_set:
            
            if example[0].split('.')[-2] == 'ham':
                label = 0
            else:
                label = 1
            
            L1_prediction = self.test_L1(example[2])
            L2_prediction = self.test_L2(example[2])
            L_inf_prediction = self.test_L_inf(example[2])
            
            if label == L1_prediction: #correct prediction
                if label == 0:
                    res_L1["correct_ham"] += 1 #true neg
                else:
                    res_L1["correct_spam"] += 1 #true pos
            else:
                if label == 0:
                    res_L1["wrong_ham"] += 1 #false neg
                else:
                    res_L1["wrong_spam"] += 1 #false pos                
            
            if label == L2_prediction:
                if label == 0:
                    res_L2["correct_ham"] += 1 #true neg
                else:
                    res_L2["correct_spam"] += 1 #true pos
            else:
                if label == 0:
                    res_L2["wrong_ham"] += 1 #false neg
                else:
                    res_L2["wrong_spam"] += 1 #false pos
                
            if label == L_inf_prediction:
                if label == 0:
                    res_Linf["correct_ham"] += 1 #true neg
                else:
                    res_Linf["correct_spam"] += 1 #true pos
            else:
                if label == 0:
                    res_Linf["wrong_ham"] += 1 #false neg
                else:
                    res_Linf["wrong_spam"] += 1 #false pos

        total = len(test_set)

        res = []

        print("-------------------------------------L-1 metric-------------------------------------")
        res.append(self.get_scores(res_L1, total))
        print("-------------------------------------L-2 metric-------------------------------------")
        res.append(self.get_scores(res_L2, total))
        print("-------------------------------------L-inf metric-------------------------------------")
        res.append(self.get_scores(res_Linf, total))

        return res
        
if __name__ == '__main__':
    
    start_time = time.process_time()
    
    print("PRE-PROCESSING DATA...")
    print("-------------------------------------")
    pre_process = PreProcess()
    folder = 'data/'
    data = pre_process.pre_process(folder)

    splits = [.9, .8, .7, .6]
    res_k1 = []
    res_k5 = []

    print("##########################################################################")
    print("################################## K=1 ##################################")
    print("##########################################################################")
    for split in splits:
        train, test = train_test_split(data, test_size = split)

        print("-------------------------------------")
        print('Training/Testing using KNN (k=1), split = {}/{}'.format(split, (1-split) ))
        print("-------------------------------------\n")
        KNN_1 = NearestNeighbors(train, 1)
        KNN_1.train(train)
        res = KNN_1.test(test)
        res_k1.append(res)

    print("##########################################################################")
    print("################################## K=5 ##################################")
    print("##########################################################################")
    for split in splits:
        train, test = train_test_split(data, test_size = split)

        print("-------------------------------------")
        print('Training/Testing using KNN (k=1), split = {}/{}'.format(split, (1-split) ))
        print("-------------------------------------\n")
        KNN_5 = NearestNeighbors(train, 5)
        KNN_5.train(train)
        res = KNN_5.test(test)
        res_k5.append(res)
    
    end_time = time.process_time()
    print("Program completed in {} seconds.".format(end_time - start_time))