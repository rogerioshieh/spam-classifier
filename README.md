# SPAM CLASSIFIER

## DESCRIPTION
This program takes a set of labelled emails (spam/ham) and uses Machine Learning algorithms that is able to classify them automatically. 

## METHODS
The program splits emails in training and test sets of various sizes to compare performances of classifiers using Naive Bayes and K-Nearest Neighbors using L1, L2, and L-inf metrics (k=1, k=5). It uses accuracy and F1 scores to measure performance and outputs a set of graphs with performance scores.

## RESULTS

![Graph 1]( /output/1.png "Graph 1" ) ![Graph 2]( /output/2.png "Graph 2" )
![Graph 3]( /output/3.png "Graph 3" ) ![Graph 4]( /output/4.png "Graph 4" )

As the above graph shows, although there were some differences about each split, the best classifiers were Naive Bayes and KNN L-2. KNN L-1 performed significantly worse. Considering that the run time of Naive Bayes is the fastest, this is the favorite method. 

## HOW TO RUN
Run `python main.py` or `python main.py --> output.txt` if you wish the results to be saved in a text file, otherwise they will be printed on the terminal.

Note that the program assumes that the txt files are already separated between spam/ham, in the format of the files in the data folder. 

## NOTES
The program is designed to run all classifiers at once and automatically plot the results. Because of that, it might take a few minutes to run in its entirety (I made sure to use the appropriate Data Structures). It is possible to run each individual classifier (Naive Bayes, KNN) if you wish to observe/test them separately. Running `python plt.py` will display a sample graph with hardcoded results. 

Author: Rogerio Shieh Barbosa
