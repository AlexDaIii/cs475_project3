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
    def predict(self, X, y):
        pass


class Adaboost(Predictor):

    def __init__(self):
        self.model = None
        self.d = None

    def train(self, x, y, boost_iter=10):
        # sort by col
        x = np.sort(x, axis=0)
        #print(x.shape)
        #print(cuts.shape)

        # find all possible cuts
        cuts = ((x[0:np.size(x, 0)-1, :] + x[1:np.size(x, 0), :]) / 2)
        # each row is the feature idx, cutoff, a, flip or not
        self.model = np.zeros((boost_iter, 4))
        # weight of instance
        # num examples, iteration
        self.d = 1/(np.size(x, 0))*np.ones((np.size(x, 0), boost_iter))

        # num features, number of cuttoffs, less than greater than
        for num_feat in range(0, np.size(x, 1)):
            min_err_for_feat = float('inf')
            alpha = 0
            best_cut = None
            flip = False

            # number of possible cuttoffs
            for num_poss_cuts in range(0, np.size(cuts, 0)):
                h = np.count_nonzero(np.greater(x[0:np.size(x, 0), num_feat], cuts[num_poss_cuts, num_feat]))

                error = self.d[num_poss_cuts, 1]*(np.count_nonzero(h != y))

                if error > 0.5:
                    flip = True
                    error = 1 - error
                if error < min_err_for_feat:
                    min_err_for_feat = error
                    best_cut = cuts[num_poss_cuts, num_feat]

            #alpha = (1/2)*math.log((1-min_err_for_feat)/min_err_for_feat)

            print()

        return cuts

    def predict(self, X, y):
        pass
