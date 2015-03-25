import random
import numpy as np
import csv
from numba import jit, float64
from scipy.special import expit
from math import log, exp, sqrt
from datetime import datetime
import threading

#calculates negative likelihood loss (NLL) over vectors of p and y
@jit(nopython = True, nogil = True)
def logloss(p, y):
    nData = len(y) #number of observations
    loss = 0.
    
    #iterates thru observations and calculates NLL
    for i in xrange(nData):
        p_curr = 1./(1.+exp(-p[i]))
        p_curr = max(min(p_curr, 1. - 10e-15), 10e-15)
        if y[i] == 1.:
            loss += -log(p_curr)
        else:
            loss += -log(1. - p_curr)
    return loss / float(nData)

#calculates NLL for a given feature and split point
@jit(nopython = True, nogil = True)
def feat_split_logloss(alpha, reg, x, y, p_current, grad, split, result):
    
    #if split value is maximum value, then return 1., so that algorithm wouldn't split on that value
    max_value = np.max(x)
    if split[0] == max_value or split[0] == 0.:
        result[:] = 1.
        return result
    
    #Get gradient for lower and upper splits
    length = float(x.shape[0])
    lower_p = 0.
    upper_p = 0.
    lower_count = 0.
    
    for m, i in enumerate(x):
        if i <= split[0]:
            lower_p += grad[m]
            lower_count += 1.
        else:
            upper_p += grad[m]
    
    upper_count = float(x.shape[0]-lower_count)
    reg_term_lower = sqrt(float(lower_count) / float(lower_count + reg))
    reg_term_upper = sqrt(upper_count / float(upper_count + reg))
    
    lower_p = lower_p / lower_count * reg_term_lower
    upper_p = upper_p / upper_count * reg_term_upper
    
    #count logloss if we split on current split value
    loss = 0.
    for m, i in enumerate(x):
        if i <= split[0]:
            loss += logloss_single(p_current[m] - lower_p * alpha, y[m])
        else:
            loss += logloss_single(p_current[m] - upper_p * alpha, y[m])
    result[:] = loss/length
    return result

#NLL over a single p/y pair
@jit(nopython = True, nogil = True)
def logloss_single(p, y):
    p = 1./(1.+exp(-p))
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)    

#main class for training a MatrixNet
class MatrixNet():
    def __init__(self, x_master, y_master, x_cv, y_cv, fraction, alpha, epoches = 1000, 
                 max_depth = 6, start_depth = 1, k_splits = 8, reg = 100):
        self.x_master = x_master #trainig features
        self.y_master = y_master #training labels
        self.x_cv = x_cv #cross-validation features
        self.y_cv = y_cv #cross-validation labels
        self.fraction = fraction #fraction of data to be used for each iteration of stochastic dbm
        self.alpha = alpha #learning rate
        self.epoches = epoches #number of iterations
        self.start_depth = start_depth #depth constraint for trees
        self.max_depth = max_depth #depth constraint for trees
        self.k_splits = k_splits #number of split points to be estimated for each feature
        self.reg = reg #regularization parameter, lambda
        self.no_feats = [] #blacklist of features that do not improve cv

    #calculates predictions for every observation in X given tree decision rules
    def classify(self, X, tree):
        predictions = np.zeros(X.shape[0]) #set up a variable to store predictions
        
        if type(tree) != list: #if current level of tree is not a list, then it's a prediction
            predictions[:] = tree
            return predictions
        
        #at current split of a tree determine variable and split point
        curr_feat = tree[0][0]
        curr_split = tree[0][1]
        
        #get indices for observations that belong to 
        lower_split = np.less_equal(X[:, curr_feat], curr_split)
        
        #recurse to the next level of a tree
        for i in xrange(2):
            # Find the datapoints with each feature value
            if i == 0: #if lower than split point
                new_X = X[lower_split,:]
                t = tree[1]
                subtree = self.classify(new_X, t)
                predictions[lower_split] = subtree
            else: #if higher than split point
                new_X = X[-lower_split,:]
                t = tree[2]
                subtree = self.classify(new_X, t)
                predictions[-lower_split] = subtree
        return predictions

    #main function to create a tree
    def make_tree(self, X, y, p_current, max_level, level = 0):
        nData = len(y) #number of observations
        nFeatures = X.shape[1] #number of features
        
        if nData == 0:
            return 0.
        elif level >= max_level: #if depth constraint is reached
            reg_term = sqrt(float(nData) / float(nData + self.reg)) #regularization in a leaf
            return np.mean(expit(p_current) - y) * reg_term * self.alpha #p-y gradient
        else:
            loss = np.zeros((nFeatures, 2)) #set up a variable to store split points and loglosses for features
            grad = np.subtract(expit(p_current), y) #calculate current gradient
            
            #calculating logloss for every feature in order to make decision where to split
            for feature in xrange(nFeatures):
                loss[feature, :] = self.calc_feat_logloss(X[:,feature], y, p_current, grad)
                
            for feat in self.no_feats: #if there is a feature that we don't want to use
                loss[feat, 0] = 1. #set it's value to 1., so that it won't be picked
            
            #finding best feature and creating a branch
            bestFeature = np.argmin(loss[:,0])
            bestFeature_name = bestFeature
            bestFeature_split = loss[bestFeature, 1]

            tree = [(bestFeature_name, bestFeature_split), [], []]

            lower_split = np.less_equal(X[:,bestFeature_name], bestFeature_split)
        
            for i in xrange(2):
                # Find the datapoints with each feature value
                if i == 0: #if lower than split point
                    new_X = X[lower_split,:]
                    new_y = y[lower_split]
                    new_p = p_current[lower_split]
                else: #if higher than split point
                    new_X = X[-lower_split,:]
                    new_y = y[-lower_split]
                    new_p = p_current[-lower_split]
                
                # Now recurse to the next level
                subtree = self.make_tree(new_X, new_y, new_p, max_level, level = level + 1)
            
                # And on returning, add the subtree on to the tree
                tree[i+1] = subtree
            return tree
        
    #technical function, used to make threads for feat_split_logloss function
    def make_multithread(self, x, y, p_current, grad, inner_func):    
        def func_mt(splitters):
            length = splitters.shape[0]
            result = np.zeros(length, dtype=np.float64)
            args = (splitters,)+(result,)
            
            c_args = [self.alpha, self.reg, x, y, p_current, grad]
            chunks = [[args[0][i:i+1],args[1][i:i+1]]  for i in xrange(length)]
            
            threads = [threading.Thread(target=inner_func, args=c_args+chunk)
                        for chunk in chunks]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return result
        return func_mt
    
    #function that coordinates finding the best feature and split point to split on
    def calc_feat_logloss(self, x, y, p_current, grad):        
        splitters = self.get_split_points(x)
        
        func_nb_mt = self.make_multithread(x, y, p_current, grad, feat_split_logloss)
        loss_master = func_nb_mt(splitters)
        
        best = np.argmin(loss_master)
        return loss_master[best], splitters[best]
    
    #gets split points given x
    def get_split_points(self, x):
        x_curr = np.sort(x) #sort x to make correct split points
        splitters = np.zeros(self.k_splits)
        nData = x_curr.shape[0]
        k = int(float(nData)/float(self.k_splits+1))
        
        index = 0
        if k == 0:
            return np.array([x_curr[0]])
        for n in xrange(k-1, nData, k):
            if index >= self.k_splits: continue
            splitters[index] = x_curr[n]
            index += 1
        return np.unique(splitters[:index])
    
    #high-level function that launches everything else
    def train(self):
        start = datetime.now()
        
        random.seed(13) #set seed for debugging and reproducibility
        trees = [] #variable to store all trees
        nData = self.y_master.shape[0] #number of observations
        
        p_cv = self.x_cv[:,-1] #use logistic regression z for initial estimations for cv data
        
        depth = self.start_depth #starting tree depth
        cv_result = [] #variable to store results for cross-validation
        cv_loss = logloss(p_cv, self.y_cv)
        cv_result.append(cv_loss)
        print('Starting cv loss:%f' % cv_loss)
        
        for iter in xrange(self.epoches):
            iter_start = datetime.now()
            
            #if iter % 10 == 0 and iter > 0: #increment current depth every 10 epoches
            #    if depth < self.max_depth:
            #        depth += 1
            
            #Getting a random fraction of data for an iteration
            ind = random.sample(xrange(nData), int(nData*self.fraction))
            x_fraction = self.x_master[ind,:]
            y_fraction = self.y_master[ind]
            p_current = x_fraction[:,-1] #use logistic regression z for initial estimations for train data
            
            for old_tree in trees:
                p_current = np.subtract(p_current, self.classify(x_fraction, old_tree))
            
            loss = logloss(p_current, y_fraction) #train loss
            
            #Main algorithm - make tree that splits best the data and make predictions
            tree = self.make_tree(x_fraction, y_fraction, p_current, depth)
            
            #cross-validation
            new_p_cv = self.classify(self.x_cv, tree)
            cv_loss = logloss(p_cv-new_p_cv, self.y_cv)
            
            print('%s\tfraction: %f\ta: %f\tdepth: %d\ttrees: %d\treg: %d\ttrain: %f\tcv: %f' 
                    % (datetime.now(), self.fraction, self.alpha, depth, len(trees), self.reg, loss, cv_loss))
            
            if cv_result[-1] - cv_loss <= 0.000001:
                self.no_feats.append(tree[0][0])
                print('no feat added:%d' % tree[0][0])
                print('prune the tree')
                if len(self.no_feats) == 20: #if all features are in blacklist, then go to next level
                    print('going to next depth level')
                    depth += 1
                    self.no_feats = []
                    if depth > self.max_depth:
                        break
            else:
                print('+1, grow the tree')
                trees.append(tree)
                cv_result.append(cv_loss)
                p_cv = np.subtract(p_cv, new_p_cv)
            
        print('Done, time elapsed: %s' % (datetime.now() - start))