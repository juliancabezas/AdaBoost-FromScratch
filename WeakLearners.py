###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assignment 2
# Weak learner sto be used in the AdaBoost libraryDecision Stumps
####################################

# Import libraries
import numpy as np
import pandas as pd


# Simple decision Stump for two class classifiers
class DecisionStump():

    # Initialization
    def __init__(self):

        # Chosen feature index
        self.chosen_feature = None

        # Threshold in the feature
        self.chosen_thr = None

        # Direction of the classification
        self.chosen_dir = None
       
        # Minimum weighted error in the classification
        self.min_err = float('Inf')

    # Train the classifier
    def fit(self, x, y, weights=None):

        # Calculate the dimensions of the X matrix
        n_rows, n_columns = x.shape

        # if the weights are not provided initialize them
        if weights is None:
            weight_first = 1 / n_rows
            weights = np.full(n_rows, weight_first)

        # Test all different features
        for feature in range(n_columns):

            # Test two different directions for each feature
            for dir in [-1,1]:

                # Test all the features values as threshold
                for thr in x[:,feature]:
                
                    # it will predict using a single feature with the threshold we are testing
                    predict = np.where(x[:,feature] >= thr, dir, -1*dir)
                    
                    # get a list with the misclassifications
                    misclassifications = predict != y

                    # calculate weighted error
                    err = sum(weights[misclassifications])

                    # If this is the minimum error update the class attributes
                    if err < self.min_err:
                        
                        self.chosen_feature = feature
                        self.chosen_thr = thr
                        self.chosen_dir = dir
                        self.min_err = err

    # Predict the class determined by the threshold and direction on the selected feature
    def predict(self,x):
        
        # Generate the predicted values
        predict = np.where(x[:,self.chosen_feature] >= self.chosen_thr, self.chosen_dir, -1*self.chosen_dir)

        return predict

