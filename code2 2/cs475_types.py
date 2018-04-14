from abc import ABCMeta, abstractmethod
import numpy as np
import math

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        # TODO
        pass
        
    def __str__(self):
        # TODO
        pass


class FeatureVector:
    def __init__(self):
        # TODO
        pass
        
    def add(self, index, value):
        # TODO
        pass
        
    def get(self, index):
        # TODO
        return 0.0
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label


# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass


class Adaboost(Predictor):

    def __init__(self):
        self.model = None
        self.d = None
        self.num_input_features = None
        self.boost_iter = None
        self.num_trees = 0

    def train(self, x, y, boost_iter=10):
        #print(x.shape[1])
        #print(x.shape[0])
        self.num_input_features = x.shape[1]
        self.boost_iter = boost_iter
        #print(x.shape)

        # sort each column
        xp = np.sort(x, axis=0)
        # find all possible cuts
        cuts = ((xp[0:np.size(x, 0)-1, :] + xp[1:np.size(x, 0), :]) / 2)
        xp = None
        #print(cuts)

        #print(np.size(cuts, 0))

        # each row is the feature idx, cutoff, a, flip or not
        self.model = np.zeros((boost_iter, 4))
        # weight of instance
        # num examples, iteration
        self.d = 1/(np.size(x, 0))*np.ones((np.size(x, 0), 1))

        # loop through the boosting algorithm
        for t in range(0, self.boost_iter):
            # need to find: best feature to cut on for that boosting iteration
            min_err_so_far = float('inf')
            best_feature = None
            best_cut = None
            best_flip = None

            # go through the whole entire cuts matrix to find best feature to cut on
            # THIS IS FITTING THE DECISION TREE
            for num_col in range(0, self.num_input_features):
                #print(num_col)
                for num_row in range(0, np.size(cuts, 0)):
                    # this is the prediction
                    pred = np.greater(x[:, num_col], cuts[num_row, num_col])
                    # if prediction does not match y, its an error
                    error = pred != y
                    # print(error)
                    # this is the epislon for that decision tree, on the current dT
                    epislon = np.dot(self.d.T, error)
                    #print(np.sum(error))

                    # check if we need to flip the prediction to less than
                    if epislon > 0.5:
                        flip = True
                        epislon = 1 - epislon
                    else:
                        flip = False
                    # save the best epsilon
                    if epislon < min_err_so_far:
                        #print(str(num_col) + " " + str(num_row))
                        best_feature = num_col
                        best_flip = flip
                        min_err_so_far = epislon
                        best_cut = cuts[num_row, num_col]

            #print(min_err_so_far)
            # AFTER GETTING THE BEST hyptothesis for that iteration
            # each in first column is the feature idx
            self.model[t, 0] = best_feature
            # , cutoff value
            self.model[t, 1] = best_cut
            # flip or not
            self.model[t, 3] = best_flip

            # CHECK IF ALPHA CAN BE CALCULATED
            if min_err_so_far < 0.000001:
                # CANNOT calculate alpha
                break
            # number of trees we can select
            self.num_trees += 1

            # ALPHA
            self.model[t, 2] = (1/2)*math.log((1-min_err_so_far)/min_err_so_far)

            # calculate d
            if self.model[t, 3]:
                error = np.logical_xor(np.less_equal(x[:, best_feature], best_cut), y)
            else:
                error = np.logical_xor(np.greater(x[:, best_feature], best_cut), y)
            for idx in range(np.size(self.d, 0)):
                # if wrong
                if error[idx] == 1:
                    self.d[idx] = self.d[idx] * np.exp(self.model[t, 2])
                # if right
                else:
                    self.d[idx] = self.d[idx] * np.exp(-self.model[t, 2])

            # self.d = np.multiply(self.d, np.exp(-alpha * np.multiply(y, best_prediction)))
            self.d = np.divide(self.d, np.sum(self.d))

        #print(self.model)
        return self.model

    def predict(self, x):
        pred = 0
        # ADD THE ALPHAS
        for idx in range(self.num_trees):
            # if flip - less equal
            if self.model[idx, 3]:
                if x[0, self.model[idx, 0]] <= self.model[idx, 1]:
                    pred += self.model[idx, 2]
                else:
                    pred -= self.model[idx, 2]
            # if not flip - greater than
            else:
                if x[0, self.model[idx, 0]] > self.model[idx, 1]:
                    pred += self.model[idx, 2]
                else:
                    pred -= self.model[idx, 2]

        if pred < 0:
            return 0
        return 1
