# AdaBoost implementation from scratch

This code implements the Adaboost Algorithm to classify a binary class target variable usign decision stumps as weak learner.

The results are compared with the Scikit-learn Adaboost and Support Vector Machine Implementations 

## Environment

This code was tested under a Linux 64 bit OS (Ubuntu 18.04 LTS), using Python 3.7.7

## How to run this code:

In order to use this code:

1. Install Miniconda or Anaconda
2. Create a environment using the requirements.yml file included in this .zip:

Open a terminal in the folder were the requirements.yml file is (Assign1-code) and run:

    ```
    conda env create -f requirements.yml --name adaboost-env
    ```

3. Make sure the folder structure of the project is as follows

```
Assign1-code
├── data
├── figures
├── results
├── 01-AdaBoost-Custom.py
├── 02-AdaBoost-Sklearn.py
├── 03-SVM-Sklearn.py
├── 04-Figutes-Adaboost-SVM.py
├── AdaBoost.py
├── WeakLearners.py
├── PerformanceMetrics.py
└── ...
```

If there are csv files in the results folder the code will read them to avoid the delay of testing different number of learners to fitting the models

5.  Run the code in the conda environment: Open a terminal in the Assign2-code folder  and run 
	```
	conda activate adaboost-env
	python 01-AdaBoost-Custom.py
    python 02-AdaBoost-Sklearn.py
    python 03-SVM-Sklearn.py
    python 04-Figutes-Adaboost-SVM.py
    ```
    
It will run the custom implementation of the Adaboost algorithm contained in a separate file (AdaBoost.py), along with creatings the figures
The code will use the Decision Strump programmed in teh WeakLearners.py file, but it can use other weak learners

Alternatevely, run the .py codes in your IDE of preference, (VS Code with the Python extension is recommended), using the root folder of the directory (Assign2-code) as working directory to make the relative paths work.

Note: You can also build your own environment following the package version contained in requirements.yml file