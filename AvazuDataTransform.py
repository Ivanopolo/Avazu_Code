import csv
from datetime import datetime
import hashlib
import os
from collections import defaultdict

class AvazuDataTransform:

    def __init__(self, path):
        self.path = path

    def transform(self, train):
        D = 2**26 #hash trick dimensions for user id
    
        #Reading train/test file
        diff = 0
        if train:
            print('Transforming train data...')
            readFile = open(self.path + 'train.csv', "rb")
        else:
            print('Transforming test data...')
            readFile = open(self.path + 'test.csv', "rb")
            diff = -1

        reader = csv.reader(readFile)
        header = next(reader)

        #Delete headers that won't be used
        del header[17+diff:19+diff] #delete columns C15-C16, merged into banner_size
        del header[5+diff:11+diff] #delete site*/app* columns, merged into placement* features
        del header[2+diff:4+diff] #delete hour and C1, separated into hour/week_day features with C1 correction

        #Create headers for new columns
        header.append('freq') #user id frequency within a single day
        header.append('user_id')
        header.append('hour')
        header.append('placement_id') #merged from site/app features
        header.append('placement_domain')
        header.append('placement_category')
        header.append('banner_size')

        #Set up directory for transformed data
        LR_path = self.path + 'LR_data\\'
        if not os.path.exists(LR_path):
            os.makedirs(LR_path)

        if train:
            #Creating files for folds
            writers = []
            for i in range(10):
                writeFile = open(LR_path + "Part%d.csv" % i, "wb")
                writers.append(csv.writer(writeFile))
                writers[i].writerow(header)
        else:
            writeFile = open(LR_path + "test.csv", "wb")
            writer = csv.writer(writeFile)
            writer.writerow(header)

        #Setting up variables for frequency / date / time data transformation
        hours_bound = 24
        freq_bound = 100
        hours_count = 0
        h_last = 0
        count = [0] * D #D-dimensional used_id frequency count

        #Reading a single row, make transformations and write it to a newly created file.
        #Newly created files are separated by different days for easier CV on different days.
        for t, row in enumerate(reader):
            curr_hour = int(row[2+diff][-2:]) % hours_bound #get current hour

            if h_last != curr_hour: #if current row's hour is different, add 1 to hour count
                hours_count += 1
            if hours_count == hours_bound: #if we reach 24 hours, then reset frequency count
                count = [0] * D
                hours_count = 0

            #Creating user_id from device_id, device_ip, device_model and C14 features
            userString = row[11+diff] + '_' + row[12+diff] + '_' + row[13+diff] + '_' + row[19+diff]
            userID = abs(hash(userString)) % D #Hash userID to D-dimensional space, used for speed

            #Update user_id frequency
            count[userID] += 1
            if count[userID] > freq_bound: #Set a cap for frequency at freq_bound
                count[userID] = freq_bound
            row.append(count[userID])

            #Hash user id to 8 symbols (for better disk spase usage)
            userID_hash = hashlib.md5()
            userID_hash.update(bytearray(userString, 'utf-8'))
            row.append(userID_hash.hexdigest()[:8])

            hour = int(row[2+diff][-2:]) #Extract time from hour feature
            timezone = int(row[3+diff][-2:]) #Extract timezome from C1
            hour = (hour + timezone) % 24
            row.append(hour)

            day = int(row[2+diff][-4:-2]) % 10

            if row[5+diff] == '85f751fd': #if site_id is NULL, then use app features
                row.append(row[8+diff])
                row.append(row[9+diff])
                row.append(row[10+diff])
            else:
                row.append(row[5+diff])
                row.append(row[6+diff])
                row.append(row[7+diff])

            row.append(row[17+diff] + '_' + row[18+diff]) #Concat banner size to single feature

            del row[17+diff:19+diff] #delete columns C15-C16, merged into banner_size
            del row[5+diff:11+diff] #delete site*/app* columns, merged into placement* features
            del row[2+diff:4+diff] #delete hour and C1, separated into hour/week_day features with C1 correction

            if train:
                writers[day].writerow(row)
            else:
                writer.writerow(row)
            if t % 10 ** 5 == 0 and t > 0: print('%s\trow: %d' % (datetime.now(), t))

            h_last = curr_hour #remember the last hour

    def count_features(self):
        #Set up variables
        count = defaultdict(int) #storage of features:count
        epoches = [0,1]
        dirFile = self.path

        '''
        Make 2 passes over data:
            - First, to collect and count all features (header + label) in test data
            - Second, to count test features in training data
        Every pass creates a file with the output for further investigation and usage
        '''
        for epoch in epoches:
            if epoch == 0: 
                print('Initial scan through data...')
                skip = [0]
                files = ['test.csv']

            if epoch == 1: 
                print('Second scan through data...')
                skip = [0, 1]
                files = ['Part%d.csv' % part for part in range(10)]
                count = dict.fromkeys(count, 0)

            tt = 1

            for file in files:
                train = dirFile + file
                header = []
                for t, line in enumerate(open(train)):
                    for m, feat in enumerate(line.rstrip().split(',')):
                        if t == 0:
                            header.append(feat)
                            continue
                        elif m in skip:
                            continue
                        elif epoch == 1: #for training data, if feature existed in test data, then it's counted
                            feat = header[m] + '_' + feat
                            if feat in count:
                                count[feat] += 1
                        else:
                            count[header[m] + '_' + feat] += 1
                    if tt % 1000000 == 0:
                        print('%s\trow: %d' % (datetime.now(), tt))
                    tt += 1

            if epoch == 0:
                filename = 'feature_names_test.csv'
                test_feats = {key for key in count.keys()}
                print('Number of test features: %d' % len(test_feats))
            if epoch == 1:
                filename = 'feature_names_train.csv'
            
            with open(dirFile + filename, 'w') as outfile:
                outfile.write('name,count\n')
                for key, value in count.items():
                    if value > 0:
                        outfile.write('%s,%d\n' % (key, value))

    def update_rare_features(self, min_bound = 0):
        '''
        update_rare_features:
        - Loads feature names that were encountered both in train and test data (freq_feats).
        - Goes through test and train data, if encounters feature not present in freq_feats, sets its value to 'none'.
        '''

        readFeats = open(self.path + 'feature_names_train.csv', 'r')
        reader = csv.reader(readFeats)
        freq_feats = []
        next(reader)

        for t, row in enumerate(reader):
            if int(row[1]) > min_bound:
                freq_feats.append(row[0])

        freq_feats = {feat for feat in freq_feats}

        print('# of Frequent Features: %d' % len(freq_feats))

        diff = 0

        for i in xrange(11):
            if i == 10:
                print('Updating rare features for test data')
                readFile = open(self.path + "test.csv", "rb")
                writeFile = open(self.path + "test_none.csv", "wb")
                diff = -1
            else:
                print('Updating rare features for train data Part%d' % i)
                readFile = open(self.path + "Part%d.csv" % i, "rb")
                writeFile = open(self.path + "Part%d_none.csv" % i, "wb")

            reader = csv.reader(readFile)
            writer = csv.writer(writeFile)

            for t, row in enumerate(reader):
                if t == 0:
                    header = row[:]
                    m = len(header)
                else:
                    for n_feat in xrange(2+diff, m):
                        feat = header[n_feat] + '_' + row[n_feat]
                        if not feat in freq_feats:
                            row[n_feat] = 'none'

                writer.writerow(row)
                if t % 10 ** 5 == 0: print('%s\trow: %d' % (datetime.now(), t))
            readFile.close()
            writeFile.close()

    def count_risk_features(self):
        '''
        count_risk_features counts clicks and impressions for all features encountered in non-cv data (1 thru 8)
        The data it then being used by gbm to form Risk Features
        '''

        dirFile = self.path
        feat_imps = defaultdict(int)
        feat_clicks = defaultdict(int)
        cv = [0, 9]

        tt = 1

        for i in xrange(10):
            if i in cv: continue
            file = 'Part%d.csv' % i
            train = dirFile + file
            print('Counting features from file %s' % file)
            header = []
            for t, line in enumerate(open(train)):
                y = 0
                for m, feat in enumerate(line.rstrip().split(',')):
                    if t == 0:
                        header.append(feat)
                        continue
                    elif m == 0:
                        continue
                    elif m == 1:
                        y = int(feat)
                    else:
                        feat = header[m] + '_' + feat
                        feat_imps[feat] += 1
                        feat_clicks[feat] += y
                if tt % 1000000 == 0:
                    print('%s\trow: %d' % (datetime.now(), tt))
                tt += 1
            
            filename = 'feats_imps_clicks.csv'
            with open(dirFile + filename, 'w') as outfile:
                outfile.write('name,imps,clicks\n')
                for key, value in feat_imps.items():
                    outfile.write('%s,%d,%d\n' % (key, value, feat_clicks[key]))

    def get_Data_Transformed(self):
        #Main function and does all high-level manipulations

        start = datetime.now()

        #Initial transform of the data
        self.transform(train = False)
        self.transform(train = True)

        self.path = self.path + 'LR_data\\'

        #Finding features that encountered in both train and test data
        self.count_features()

        #Leaving only features that were determinged in count_features function
        self.update_rare_features()

        #Get clicks / impression counts for all features (to be used in GBM)
        self.count_risk_features()

        print('Done, elapsed time: %s' % str(datetime.now() - start))

clTransform = AvazuDataTransform('C:\\Users\\User\\Desktop\\Avazu\\Avazu_habrahabr\\')
clTransform.get_Data_Transformed()

###Time to run: ~23 min
###Disk space: 5.5gb