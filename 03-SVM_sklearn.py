###################################
# Julian Cabezas Pena
# Introduction to Statistical Machine Learning
# University of Adelaide
# Assingment 2
# Support Vector Machine Classifer using scikit-learn
####################################

# Import standard libraries
import numpy as np # Numerical calculations
import pandas as pd # read csv dataset
import os
import time

# Custom implementation of the accuracy calculation
from PerformanceMetrics import accuracy

# Adaboost classfier in SkLearn package
from sklearn.svm import SVC

# Main function, reads data, test different cost and kernel and outputs the best model errors
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

    # Test different maximum number of features in each split
    cost_array = np.arange(start=0.5, stop=10.5, step=0.5)
    cost_array = np.insert(cost_array , 0, 0.1, axis=0)

    # Test different numbers of trees from 10 to 550
    kernel_array = ['linear', 'poly', 'rbf']

    # Store the partial results in lists
    cost_list = []
    kernel_list = []
    err_train_list = []
    err_test_list = []

    # Check if the tuning was already done
    if not os.path.exists('results/cost_kernel_svm_sklearn.csv'):

        # Loop trough the different combinations cost and kernel
        for cost in cost_array:

            for kernel in kernel_array:
                
                print("Testing the model with cost = ", cost)
                print("Testing the model with kernel = ", kernel)

                # Initialize the Adaboost class with the correspoinding number of learners
                svm = SVC(kernel = kernel, C = cost)

                # Fit the model on the training data
                svm.fit(x_train, y_train)

                # Predict on the training data and get the error
                prediction_train = svm.predict(x_train)
                acc_train = accuracy(ytrue = y_train, ypred = prediction_train)

                # Predict on the test data and get the error
                prediction_test = svm.predict(x_test)
                acc_test = accuracy(ytrue = y_test, ypred = prediction_test)

                # Store the results in the lists
                cost_list.append(cost)
                kernel_list.append(kernel)
                err_train_list.append(1.0 - acc_train)
                err_test_list.append(1.0 - acc_test)

        # Create pandas dataset and store the results
        dic = {'cost':cost_list,'kernel':kernel_list, 'error_train':err_train_list,'error_test':err_test_list}
        df_results = pd.DataFrame(dic)
        df_results.to_csv('results/cost_kernel_svm_sklearn.csv')
        print("Testing of different cost and kernel ready!")
    else:
        # In case the parameters were already tuned, read the results
        df_results= pd.read_csv('results/cost_kernel_svm_sklearn.csv')
        print("Previous testing of cost and kernel detected")


    #----------------------------------------------
    # Fitting model
    print("")
    print("Step 3: Best Model fitting and prediction")
    print("")

    # Search the minimum error index in the dataframe
    row_max = df_results['error_test'].argmin()

    # Get the the better cost parameter
    cost = float(df_results['cost'].values[row_max])

    # Get the the better kernel parameter
    kernel = df_results['kernel'].values[row_max]

    # Initialize the Adaboost class
    svm = SVC(kernel = kernel, C = cost)
    
    print("Fitting model with cost = ", cost, "and",kernel, "kernel...")

    # Get the start time of the fitting
    fit_starts = time.time()

    # Fit the model on the training data
    svm.fit(x_train, y_train)

    # get the time when the fitting ends
    fit_ends = time.time()

    # Print the training time
    print("Total time taken to train the model: ", fit_ends - fit_starts)

    # predict on the training data and get the error results
    prediction_train = svm.predict(x_train)
    acc_train = accuracy(ytrue = y_train, ypred = prediction_train)
    print("Train error:", 1.0 - acc_train)

    # Predict on the test data and get the error value
    prediction_test = svm.predict(x_test)
    acc_test = accuracy(ytrue = y_test, ypred = prediction_test)
    print("Test Error:", 1.0 - acc_test)

if __name__ == '__main__':
	main()



