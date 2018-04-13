from data import load_data
import numpy as np
import cs475_types

DATASETS = ['datasets/easy.train'] #, 'datasets/hard.train', 'datasets/bio.train', 'datasets/finance.train',
            #'datasets/nlp.train', 'datasets/speech.train', 'datasets/vision.train']

for filename in DATASETS:
    X, y = load_data(filename)
    ada = cs475_types.Adaboost()
    X = X.todense()
    y = y.reshape(len(y), 1)
    #X = np.array([[4,3,7],[0,1,-1],[9,14,2]])
    ada.train(X, y)
