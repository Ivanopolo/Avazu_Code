###Initial data transformation
import AvazuDataTransform
path = 'dir' ###directory to where you store Avazu data
clTransform = AvazuDataTransform(path)
clTransform.get_Data_Transformed()

###Logistic Regression algorithm
import LogisticRegression
path = 'dir' + 'LR_data\\' ###directory to where you store Avazu data
clLR = LogisticRegression(path, 
                          alpha = 0.0003, #learning rate
                          n_passes = 6, #number of epoches
                          poly = True, #to use or not 2nd order polynomial features
                          wTx=True, #at the end of run wheather to create or not data for MatrixNet to use
                          adagrad_start = 5) #when to start Adaptive Gradient
clLR.main()

###Function to load all data for MatrixNet into main memory
import MatrixNetData
data = MatrixNetData.MatrixNetData(path, ###directory to where you store Avazu data
                                   0.1613) ###default p value for features that were not encountered in test data, it's an average CTR for test data
x_train, y_train, x_cv, y_cv = data.get_data()

###MatrixNet algorithm
import MatrixNet
mn = MatrixNet(x_train, y_train, x_cv, y_cv, fraction = 0.1, alpha = .3)
mn.train()
