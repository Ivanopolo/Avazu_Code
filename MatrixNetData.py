from datetime import datetime
from collections import defaultdict
import numpy as np
import csv

class MatrixNetData():
    def __init__(self, path, default_value, test = False):
        self.path = path
        self.default_value = default_value
        self.test = test

    def get_risk_feats(self, smooth_clicks, smooth_imps):
        '''
        function get_risk_feats reads names of all features from the file
        and calculates smoothed CTR
        '''

        readFile = open(self.path + 'feats_imps_clicks.csv', 'rb')
        reader = csv.reader(readFile)
        risk_feats = defaultdict(float)

        next(reader)
        for t, row in enumerate(reader):
            key = row[0]
            imps = float(row[1]) + smooth_imps
            clicks = float(row[2]) + smooth_clicks
            risk_value = clicks / imps
            risk_feats[key] = risk_value
    
        print('Number of feats: %d' % len(risk_feats))

        return risk_feats

    def data(self, risk_feats, cv_part, traindata=False):
        diff = 0

        n_feats = 19+1 #number of features + wTx

        if traindata:
            estimated_lenght = 35 * 10 ** 6
        else:
            estimated_lenght = 9 * 10 ** 6

        x_master = np.zeros((estimated_lenght, n_feats))
        y_master = np.zeros(estimated_lenght)

        length = 0
        for i in xrange(10):
            if traindata and i in cv_part: continue
            elif not traindata and not i in cv_part: continue

            filename = 'Part%d_none.csv' % i
            file = open(self.path + filename)
            file_wTx = open(self.path + 'wTx_' + filename)
            line = next(file_wTx)
            line = next(file)
            header = line.rstrip().split(',')
            print('Getting data from %s' % filename)

            for t, line in enumerate(file):
                if t % 1000000 == 0 and t > 0:
                    print('%s\trow:%d' % (datetime.now(), t))

                for m, feat in enumerate(line.rstrip().split(',')):
                    if m == 0:
                        continue
                    elif m == 1:
                        y_master[length] = float(feat)
                    else:
                        if feat == 'none':
                            x_master[length, m-2+diff] = self.default_value
                        else:
                            feat = header[m] + '_' + feat
                            x_master[length, m-2+diff] = risk_feats[feat]
                wTx = float(next(file_wTx))
                x_master[length, -1] = wTx
                length += 1
        return x_master[:length,:], y_master[:length]

    def get_data(self):
        start = datetime.now()
        cv_part = [0,9]
        risk_feats = self.get_risk_feats(0.35, 10.)

        x_cv = []
        y_cv = []
        x_cv, y_cv = self.data(risk_feats, cv_part, traindata = False)
        x_master, y_master = self.data(risk_feats, cv_part, traindata = True)

        print('Done, time elapsed: %s' % (datetime.now()-start))

        return x_master, y_master, x_cv, y_cv

#load = load_Risk_Features('C:\\Users\\User\\Desktop\\Avazu\\Avazu_habrahabr\\LR_data\\', 0.1613)
#load.get_data()