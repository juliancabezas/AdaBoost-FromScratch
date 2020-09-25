
###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assignment 2
# AdaBooost Classifer using Decision Stumps
####################################


# Import libraries
import numpy as np
import pandas as pd
import os

from WeakLearners import DecisionStump



class Adaboost:

    def __init__(self,n_learners=10):

        # Make a list to store the weak learners
        self.learners = []

        # Make a list to store the weights of each the weak learner
        self.weights_learners = []

        # Number of learners
        self.n_learners = n_learners 


    def fit(self,x,y,):

        n_rows, n_columns = x.shape

        # Initialize the weights for the first iteration
        weights = np.full(n_samples, (1 / n_rows)) 

        for i in range(n_learners):

            # Create a decision stump 
            decision_stump = DecisionStump()

            # Fit the decision stump using the weights
            decision_stump.fit(x = x, y = y, weights = weights)

        

        




#----------------------------------------------
# Step 0: Data reading and preprocessing
print("Step 0: Data reading and preprocessing")
print("")

# Train data

# Read the breast cancer database using pandas
data = pd.read_csv("./data/wdbc_data.csv", header = None)

# Drop the ID column
data = data.drop(data.columns[0], axis=1)

# Recode the output column to get -1 and 1 output values
data.iloc[:, 0] = np.where(data.iloc[:, 0] == 'M', 1, data.iloc[:, 0])
data.iloc[:, 0] = np.where(data.iloc[:, 0] == 'B', -1, data.iloc[:, 0])

# Convert to numpy array
y_full = data.iloc[:,0]
x_full = data.drop(data.columns[0], axis=1)

# Get training data
y_train = y_full.iloc[:300]
x_train = x_full.iloc[:300,:]

# Get testing data
y_test = y_full.iloc[300:]
x_test = x_full.iloc[300:,:]


#----------------------------------------------
# Fitting model
print("Step 1: Model fitting")
print("")

# Create a decision stump 
decision_stump = DecisionStump()

# Fit the decision stump using the weights
decision_stump.fit(x = x_train, y = y_train)

prediction = decision_stump.predict(x = x_train)

print(prediction)

print(prediction == y_train)

print("")


