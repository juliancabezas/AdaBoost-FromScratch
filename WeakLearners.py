# Import libraries
import numpy as np
import pandas as pd

class DecisionStump():

    # Initialization
    def __init__(self):

        # Chosen feature index
        self.chosen_feature = 9999999

        # Threshold in the feature
        self.chosen_thr = 9999999

        self.chosen_pol = 9999999

    def fit(self, x, y, weights=None):

        # Calculate the dimensions of the X matrix
        n_rows, n_columns = x.shape

        # if the weights are not provided
        if weights is None:
            weights = np.full(n_rows, (1 / n_rows))
        
        # Set the minimum error to a very large number that will be replaced
        min_err = 9999999

        # Test all different features
        for feature in x.columns:

            # Test two different polarities for each feature
            for pol in [-1,1]:

                # Test all the features values as threshold
                for values in x[feature].values:

                    thr = values
                
                    # it will predict using a single feature with the threshold we are testing
                    predict = np.where(x[feature] >= thr, pol, -1*pol)
                    
                    # get a list with the misclassifications
                    misclassifications = predict != y

                    # calculate weighted error
                    err = sum(weights[misclassifications])

                    # If this is the minimum error update the class attributes
                    if err < min_err:

                        self.chosen_feature = feature
                        self.chosen_thr = thr
                        self.chosen_pol = pol
                        min_err = err
                        print('Minimum weighted error:',min_err)

    def predict(self,x):

        predict = np.where(x[self.chosen_feature] >= self.chosen_thr, self.chosen_pol, -1*self.chosen_pol)

        return predict

