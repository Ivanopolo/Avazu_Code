from datetime import datetime
from math import log, exp, sqrt
import csv

class AvazuLR():

    #Initialize class with learning parameters
    def __init__(self, path, alpha, n_passes, poly, wTx, adagrad_start):
        self.path = path
        self.D = 2 ** 26 #Dimensions for feature hashing
        self.alpha = alpha #Learning rate
        self.n_passes = n_passes #Number of passes / epoches over the data
        self.n = [0.] * self.D #Empty vector for AdaGrad coefficients
        self.w = [0.] * self.D #Empty vector for weights
        self.poly = poly #Use/not use 2nd order polynomial features
        self.wTx = wTx #create wTx for GBM at the end of training
        self.adagrad = False #using regular gradient update for beginning
        self.adagrad_start = adagrad_start #n of pass where to start AdaGrad

    #Function for reading data line by line, hashing features and getting polynomials
    def data(self, f_name):
        file = open(self.path + f_name)
        line = next(file)
        diff = 0

        header = line.rstrip().split(',')
        n_feats = len(header)-2+diff

        for t, line in enumerate(file):
            if self.poly:
                x = [0] * int(n_feats+(n_feats)*(n_feats-1)/2)
            else:
                x = [0] * n_feats
            for m, feat in enumerate(line.rstrip().split(',')):
                if m == 0:
                    ID = int(feat)
                elif m == 1:
                    y = float(feat)
                else:
                    if not feat == 'none':
                        feat = header[m] + '_' + feat
                        feat = abs(hash(feat)) % self.D
                        x[m-2+diff] = feat

                    if m == (n_feats+1-diff) and self.poly:
                        #second order polynomial features
                        for left in xrange(n_feats):
                            if x[left] == 0:
                                continue
                            for right in xrange(1, n_feats):
                                if x[right] == 0:
                                    continue
                                elif left < right:
                                    index = (n_feats*2+1-(left+1))*(left+1)/2+right-(left+1)
                                    feat = (x[left] + x[right]) % self.D
                                    x[index] = feat
            yield (x, y)
    
    #Negative log likelihood function
    def logloss(self, p, y):
        p = max(min(p, 1. - 10e-15), 10e-15)
        return -log(p) if y == 1. else -log(1. - p) #the result for single value equals to -y*log(p)-(1-y)*log(1-p)

    #Predict probability of click given x
    def predict(self, x):
        wTx = self.w[0] #bias term, always at index 0

        for i in x:  # do wTx
            if i == 0: continue #skip features with 'none' values
            wTx += self.w[i] #equals to wTx for sparse data

        #Function return bounded sigmoid and wTx (to be used for regularization)
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.))), wTx

    #Count gradient and updates weights
    def update(self, x, p, y):
        g = p - y #get gradient for sigmoid function

        #bias
        if self.adagrad:
            self.n[0] += g ** 2 #update running sum of squared gradients (AdaGrad)
            self.w[0] -= self.alpha / sqrt(self.n[0]) * g #update bias
        else:
            self.w[0] -= self.alpha * g #update bias

        #all other features
        for i in x:
            if i == 0: continue #skip features with 'none' values
            if self.adagrad:
                self.n[i] += g ** 2
                self.w[i] -= self.alpha / sqrt(self.n[i]) * g
            else:
                self.w[i] -= self.alpha * g

    #Main train function
    def train_w(self, f_name, loss, tt, cv = False):
        '''
        Main training function:
        - Gets input values (x) and target value (y) for the current row
        - Gets prediction (p) given x
        - Updates weights (w) if it's training data
        - Updates current Negative Log Likelihood running sum (loss)
        '''
        for x, y in self.data(f_name):
            p, wTx = self.predict(x)

            if not cv:
                self.update(x, p, y)
            loss += self.logloss(p, y)

            if tt % 1000000 == 0:
                print('%s\trow: %d\tlogloss: %f' % (
                        datetime.now(), tt, (loss/tt)))

            tt += 1
        return loss, tt

    #Environment wrapper for train_w function
    def experiment(self):
        '''
        experiment function:
            - Creates a result log file
            - Launches train_w for all train files
            - Makes CV estimation based on days 0 and 9
        '''

        writeFile = open(self.path + "Experiment_a=%f.csv" % (self.alpha), "wb")
        writer = csv.writer(writeFile)
        header = ['alpha', 'empty_bins', 'pass', 'train loss', 'cv loss']
        writer.writerow(header)

        results = {'alpha':[],
                    'pass':[],
                    'empty_bins':[], 
                    'train loss': [], 
                    'cv loss': []}

        cv_part = [0, 9]

        for n_pass in xrange(self.n_passes):
            loss = 0.
            tt = 1

            # training
            for i in xrange(10):
                if i in cv_part: continue
                f_train = 'Part%d_none.csv' % i
                print('Learning from the file ' + f_train)
                loss, tt = self.train_w(f_train, loss, tt)
                
            # cross-validating
            cv_tt = 1
            cv_loss = 0.

            for i in cv_part:
                f_cv = 'Part%d_none.csv' % i
                print('CV on the file ' + f_cv)
                cv_loss, cv_tt = self.train_w(f_cv, cv_loss, cv_tt, cv = True)
                    

            empty_bins = sum(1 for i in self.w if i == 0)
            results['alpha'].append(self.alpha)
            results['empty_bins'].append(empty_bins)
            results['pass'].append(n_pass)
            results['train loss'].append(loss/tt)
            results['cv loss'].append(cv_loss/cv_tt)
            print(results)
            writer.writerow([self.alpha, empty_bins, n_pass, loss/tt, cv_loss/cv_tt])

            if n_pass == self.adagrad_start-1:
                self.adagrad = True
                print('AdaGrad is turned on!')
            
            #if it's the last pass, then make files with wTx for GBM to use
            if n_pass == self.n_passes-1 and self.wTx:
                for i in xrange(10):
                    f_train = 'Part%d_none.csv' % i
                    with open(self.path + 'wTx_' + f_train, 'wb') as outfile:
                        outfile.write('wTx\n')
                        for x, y in self.data(f_train):
                            p, wTx = self.predict(x)
                            outfile.write('%s\n' % (str(wTx)))

    #High-level function to launch experiment
    def main(self):
        start = datetime.now()

        self.experiment()

        print('Done, elapsed time: %s' % str(datetime.now() - start))

clLR = AvazuLR('C:\\Users\\User\\Desktop\\Avazu\\Avazu_habrahabr\\LR_data\\', 
                          alpha = 0.0003, n_passes = 6, poly = True, wTx=True, adagrad_start = 5)
clLR.main()