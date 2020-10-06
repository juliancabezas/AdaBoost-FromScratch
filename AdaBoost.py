###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assignment 2
# AdaBooost Classifer
####################################

# Import standard libraries
import numpy as np # Numerical calculations
import copy

# Custom implementation of the Decision Stump weak learner
from WeakLearners import DecisionStump

# Adaboost class, it works plugging any weak learner with the arguments x, y and weights
class Adaboost:

    # Parameters
    # x = explanatory variable numpy matrix
    # y = true label numpy matrix
    # n_learners; Number of weak learners (iterations to use)
    # weak_learner: classfifier function to use as weak learner, it requieres the parameters x, y and weights, along with a min_error attribute

    # Constructor, defaults to using decision stumps and 10 iterations, 
    def __init__(self,n_learners=10, weak_learner = DecisionStump()):

        # Number of learners
        self.n_learners = n_learners 

        # The weak learner to use, by default it is the Decision Stump
        self.weak_learner = weak_learner

        # Make a list to store the weak learners
        self.learners = []

        # Make a list to store the walphas of each the weak learner
        self.alphas_learners = []


    # Train the AdaBoost model
    def fit(self,x,y):

        # Get the dimensions of the x dataset
        n_rows, n_columns = x.shape

        # Initialize the weights for the first iteration
        weight_first = 1 / n_rows
        weights = np.full(n_rows, weight_first)

        # Build the learners
        for i in range(self.n_learners):

            # Create a decision stump, use the copy function to reset the learner to default values
            weak_learner = copy.copy(self.weak_learner)

            # Fit the learner using the weights
            weak_learner.fit(x = x, y = y, weights = weights)

            # Generate prediction
            prediction = weak_learner.predict(x = x)

            # Now calculate the alpha using the error in the learner
            alpha = 0.5 * np.log((1.0 - weak_learner.min_err) / weak_learner.min_err)

            # Store the alpha and the learner in lists
            self.alphas_learners.append(alpha)
            self.learners.append(weak_learner)

            # Update the weights for the next iteration
            weights = weights * np.exp(-alpha * y * prediction)

            # Now apply the normalization factor, it will leaave the sum of weights to 1
            weights = weights / np.sum(weights)

    # Predict on new data
    def predict(self,x):

        # Get the dimensions of the x dataset
        n_rows, n_columns = x.shape

        # Initialize the weighted sum of the partial learner operations
        weighted_sum = np.zeros(n_rows)
        counter = 0

        # Go through the weak learners
        for learner in self.learners:

            # Get the prediction of singular learner
            prediction = learner.predict(x)

            # Get the corresponding alpha (weight)
            alpha = self.alphas_learners[counter]

            # Calculate the sum of the predictions weighted by the alphas
            weighted_sum = weighted_sum + (alpha * prediction)

            counter = counter + 1

        # Final prediction using the sign function on the weighted sum of all the learners
        prediction_final = np.sign(weighted_sum)
        return prediction_final