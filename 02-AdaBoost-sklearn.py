
###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assignment 2
# AdaBooost Classifer using Decision Stumps
####################################


# Import standard libraries
import numpy as np # Numerical calculations
import pandas as pd # read csv dataset
import os
import time

#Custom implementation of the Decision Stump weak learner
from WeakLearners import DecisionStump

# Custom implementation of the accuracy calculation
from PerformanceMetrics import accuracy

# Adaboost classfier in SkLearn package
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Main function, reads data, test different number of predictors and outputs the best model errors
def main():

    #----------------------------------------------
    # Step 1: Data reading and preprocessing
    print("Step 1: Data reading and preprocessing")
    print("")

    # Read the breast cancer database using pandas
    data = pd.read_csv("./data/wdbc_data.csv", header = None)

    # Drop the ID column
    data = data.drop(data.columns[0], axis=1)

    # Recode the output column to get -1 and 1 output values
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 'M', 1, data.iloc[:, 0])
    data.iloc[:, 0] = np.where(data.iloc[:, 0] == 'B', -1, data.iloc[:, 0])

    # Convert to numpy array
    y_full = data.iloc[:,0].values
    y_full = y_full.astype(int)
    x_full = data.drop(data.columns[0], axis=1).values

    # Get training data
    y_train = y_full[:300]
    x_train = x_full[:300,:]

    # Get testing data
    y_test = y_full[300:]
    x_test = x_full[300:,:]


    #----------------------------------------------
    # Testing different number of learners
    print("Step 2: Testing different numbers of learners")
    print("")

    # Test different number of learners in the test and train data values in each split
    n_learners_array = np.arange(start=10, stop=510, step=10)
    n_learners_array = np.insert(n_learners_array , 0, 1, axis=0)

    # Store the partial results in lists
    n_learners_list = []
    err_train_list = []
    err_test_list = []
    time_list = []

    # Check if the tuning was already done
    if not os.path.exists('results/n_learners_adaboost_sklearn.csv'):

        # Loop trough the different combinations of step and number of iterations
        for n_learners in n_learners_array:
            
            print("Testing the model with n_learners = ", n_learners)

            # Initialize the Adaboost class with the correspoinding number of learners
            adaboost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1), n_estimators = n_learners,algorithm='SAMME')

            # Get the start time of the fitting
            fit_starts = time.time()

            # Fit the model on the training data
            adaboost.fit(x_train, y_train)

            # get the time when the fitting ends
            fit_ends = time.time()

            # Predict on the training data and get the error
            prediction_train = adaboost.predict(x_train)
            acc_train = accuracy(ytrue = y_train, ypred = prediction_train)

            # Predict on the test data and get the error
            prediction_test = adaboost.predict(x_test)
            acc_test = accuracy(ytrue = y_test, ypred = prediction_test)

            # Store the results in the lists
            n_learners_list.append(n_learners)
            err_train_list.append(1.0 - acc_train)
            err_test_list.append(1.0 - acc_test)
            time_list.append(fit_ends-fit_starts)

        # Create pandas dataset and store the results
        dic = {'n_learners':n_learners_list, 'error_train':err_train_list,'error_test':err_test_list,'train_time':time_list}
        df_results = pd.DataFrame(dic)
        df_results.to_csv('results/n_learners_adaboost_sklearn.csv')
        print("Testing of different number of learners ready!")
    else:
        # In case the parameters were already tuned, read the results
        df_results = pd.read_csv('results/n_learners_adaboost_sklearn.csv')
        print("Previous testing of different number of learners detected")

    #----------------------------------------------
    # Fitting model
    print("")
    print("Step 3: Best Model fitting and prediction")
    print("")

    # Search the minimum error index in the dataframe
    row_max = df_results['error_test'].argmin()

    # Get the the better number of learners
    n_learners = int(df_results['n_learners'].values[row_max])

    # Initialize the Adaboost class
    adaboost = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1), n_estimators = n_learners, algorithm='SAMME')
    
    print("Fitting model with", n_learners, "weak learners...")

    # Get the start time of the fitting
    fit_starts = time.time()

    # Fit the model on the training data
    adaboost.fit(x_train, y_train)

    # get the time when the fitting ends
    fit_ends = time.time()

    # Print the training time
    print("Total time taken to train the model: ", fit_ends - fit_starts)

    # predict on the training data and get the error results
    prediction_train = adaboost.predict(x_train)
    acc_train = accuracy(ytrue = y_train, ypred = prediction_train)
    print("Train error:", 1.0 - acc_train)

    # Predict on the test data and get the error value
    prediction_test = adaboost.predict(x_test)
    acc_test = accuracy(ytrue = y_test, ypred = prediction_test)
    print("Test Error:", 1.0 - acc_test)

if __name__ == '__main__':
	main()

