#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Rogerio Shieh Barbosa

"""

import pandas as pd 
import numpy as np
import time
from sklearn.model_selection import train_test_split
from naive_bayes_RSB import NaiveBayes
from KNN import NearestNeighbors
from pre_processing import PreProcess
from plt import Plot

def naive_bayes(train, test, split): #-> List(float)

    print("-------------------------------------")
    print('Naive Bayes')
    print("-------------------------------------\n")
    
    nb = NaiveBayes(np.unique([0,1]))
    nb.train(train)
    res = nb.test(test) #[total, correct, accuracy, precision, recall, f1_score]

    return res

def knn1(train, test, split): # -> List(float)

    print("-------------------------------------")
    print('KNN (k=1)')
    print("-------------------------------------\n")
    
    KNN_1 = NearestNeighbors(train, 1)
    KNN_1.train(train)
    res = KNN_1.test(test)

    return res

def knn5(train,test, split): # -> List(float)

    print("-------------------------------------")
    print('KNN (k=5)')
    print("-------------------------------------\n")
    KNN_5 = NearestNeighbors(train, 5)
    KNN_5.train(train)
    res_k5 = KNN_5.test(test)

    return res_k5

if __name__ == '__main__':
    
    start_time = time.process_time()

    print("PRE-PROCESSING DATA...")
    print("-------------------------------------\n")
    
    pre_process = PreProcess()
    folder = 'data/'
    data = pre_process.pre_process(folder)

    splits = [.9, .8, .7, .6]
    
    '''
    The following lists will contain lists, each with a result, in the format:
    [total, correct, accuracy, precision, recall, f1_score]
    '''
    nb_results = [] 
    knn1_results = []
    knn5_results = []

    for split in splits:
        print("----------------------")
        print("|| SPLIT = {}/{} ||".format(split, round(1-split, 1) ))
        print("----------------------\n")
        train, test = train_test_split(data, test_size = split)
        
        nb_res = naive_bayes(train, test, split)
        nb_results.append(nb_res)

        knn1_res = knn1(train, test, split) #[ [L1-results], [L2-results], [Linf-results]]
        knn1_results.append(knn1_res)

        knn5_res = knn5(train, test, split)
        knn5_results.append(knn5_res)
        
    end_time = time.process_time()
    print("Program completed in {} seconds ({} minutes).".format(end_time - start_time, (end_time - start_time) /60 ))

    plotting = Plot(splits, nb_results, knn1_results, knn5_results)
    plotting.plot_all()
