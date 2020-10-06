 
###################################
# Julian Cabezas Pena
# Deep Learning Fundamentals
# University of Adelaide
# Assingment 2
# Figures of the Adaboost train and testing error per number of learners
####################################

# Import the libraries to do graphs
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os

# PCA
from sklearn import decomposition

# Main functions, makes and outputs the graphs in the ./figures folder
def main():
        
    # Adaboost number of learners graphs

    # Set the style of the seaborn graphs
    sns.set_style("whitegrid")

    # Read the datasets
    adaboost_custom = pd.read_csv('./results/n_learners_adaboost_custom.csv')
    adaboost_sklearn = pd.read_csv('./results/n_learners_adaboost_sklearn.csv')

    # Melt the columns
    adaboost_custom_melt = pd.melt(adaboost_custom[['n_learners','error_train','error_test']], id_vars= ['n_learners'],value_vars=['error_train','error_test'])
    adaboost_sklearn_melt = pd.melt(adaboost_sklearn[['n_learners','error_train','error_test']], id_vars= ['n_learners'],value_vars=['error_train','error_test'])

    # Rename variables for the graph
    adaboost_custom_melt = adaboost_custom_melt.rename(columns={'variable': 'Error'},)
    adaboost_sklearn_melt = adaboost_sklearn_melt.rename(columns={'variable': 'Error'},)
    adaboost_custom_melt['Error'] = adaboost_custom_melt['Error'].replace('error_train','Train').replace('error_test','Test')
    adaboost_sklearn_melt['Error'] = adaboost_sklearn_melt['Error'].replace('error_train','Train').replace('error_test','Test')
    adaboost_custom_melt['value'] = adaboost_custom_melt['value'] * 100.0
    adaboost_sklearn_melt['value'] = adaboost_sklearn_melt['value'] * 100.0

    # Make a plot with training and test error for the custom implementation
    ax1 = sns.lineplot(x = "n_learners", y = "value", hue = "Error", data = adaboost_custom_melt).set(ylabel='Error (%)', xlabel='Number of learners (iterations)', title="Custom AdaBoost")
    plt.savefig('./figures/nlearners_adaboost_custom.pdf',bbox_inches='tight')

    # Clear the matplotlib plot environment
    plt.clf()

    # Make a plot with training and test error for the sklearn implementation
    ax2 = sns.lineplot(x = "n_learners", y = "value", hue = "Error", data = adaboost_sklearn_melt).set(ylabel='Error (%)', xlabel='Number of learners (iterations)', title="Sklearn AdaBoost")
    plt.savefig('./figures/nlearners_adaboost_sklearn.pdf',bbox_inches='tight')

    # Clear the matplotlib plot environment
    plt.clf()

    # AdaBoost training time graphs

    # Make a plot with training time for the custom implementation
    ax1 = sns.lineplot(x = "n_learners", y = "train_time", data = adaboost_custom).set(ylabel='Training time (seconds)', xlabel='Number of learners (iterations)', title="Custom AdaBoost")
    plt.savefig('./figures/train_time_custom.pdf',bbox_inches='tight')

    # Clear the matplotlib plot environment
    plt.clf()

    # Make a plot with training time for the custom implementation
    ax1 = sns.lineplot(x = "n_learners", y = "train_time", data = adaboost_sklearn).set(ylabel='Training time (seconds)', xlabel='Number of learners (iterations)', title="Custom AdaBoost")
    plt.savefig('./figures/train_time_sklearn.pdf',bbox_inches='tight')

    # Clear the matplotlib plot environment
    plt.clf()


    #--------------------------------------------------
    # Support vector machine graph

    # Read the datasets
    svm_results = pd.read_csv('results/cost_kernel_svm_sklearn.csv')
    svm_results

    # Melt the columns
    svm_results_melt = pd.melt(svm_results[['cost','kernel','error_train','error_test']], id_vars= ['cost','kernel'],value_vars=['error_train','error_test'])

    # Rename variables
    svm_results_melt = svm_results_melt.rename(columns={'variable': 'Error'},)
    svm_results_melt['Error'] = svm_results_melt['Error'].replace('error_train','Train').replace('error_test','Test')

    # Rename variables (kernel)
    svm_results_melt = svm_results_melt.rename(columns={'kernel': 'Kernel'})
    svm_results_melt['Kernel'] = svm_results_melt['Kernel'].replace('linear','Linear').replace('poly','Polynomial (degree=3)').replace('rbf','Radial')

    # Error as Percentage
    svm_results_melt['value'] = svm_results_melt['value'] * 100.0

    # Make graph and save as pdf
    ax3 = sns.lineplot(x="cost", y="value",style="Error", hue="Kernel",data=svm_results_melt).set(ylabel='Error (%)',xlabel='Cost',title="Sklearn SVM",ylim =(0,17))
    plt.savefig('./figures/cost_kernel_svm.pdf',bbox_inches='tight')

     # Clear the matplotlib plot environment
    plt.clf()

    #------------------------------------------ 
    #  PCA plot

    # Read the breast cancer database using pandas
    data = pd.read_csv("./data/wdbc_data.csv", header = None)

    # Drop the ID column
    data = data.drop(data.columns[0], axis=1)

    # Convert to numpy array
    y_full = data.iloc[:,0].values
    x_full = data.drop(data.columns[0], axis=1).values

    # Perform PCA with 2 components
    pca = decomposition.PCA(n_components=2)
    pc = pca.fit_transform(x_full)

    # Make dataframe for plotting
    pc_df = pd.DataFrame(data = pc , columns = ['Principal Component 1', 'Principal Component 2'])
    pc_df['Cancer'] = y_full

    # Make the scatterplot
    sns.lmplot( x="Principal Component 1", y="Principal Component 2", data=pc_df,  fit_reg=False, hue='Cancer', legend=True, scatter_kws={"s": 3})
    plt.savefig('./figures/pca.pdf',bbox_inches='tight')

if __name__ == '__main__':
	main()
