class DecisionStump():

    # Initialization
    def __init__(self):

        # Chosen feature index
        self.chosen_feature_idx = 9999999

        # Threshold in the feature
        self.chosen_thr = 9999999

    def fit(self,x,y,w):

        n_samples, n_features = x.size

        for feature in x.columns:

            for values in x[feature].values:

                thr = values
            
                # it will predict using a single feature with the threshold we are testing
                predict =  = np.where(x[feature] >= thr, 1, -1)



        # feature 


