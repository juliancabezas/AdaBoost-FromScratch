###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assignment 2
# Performance metrics: Accuracy
####################################

# Import standard libraries
import numpy as np # Numerical calculations
import pandas as pd # read csv dataset

# Calculate the accuracy given a true and predicted y value
def accuracy(ytrue,ypred):

    # calculate the number of samples
    n_samples = len(ytrue)

    # Sum the number of correct labels
    ones = np.ones(n_samples)
    sum_correct = np.sum(ones[ytrue == ypred])
    
    # Calculate the accuracy
    acc = (sum_correct / n_samples) *1.0

    return acc

