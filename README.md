# Avazu_Code
Code used for Avazu Kaggle Competition (Logistic Regresson, GBM, MatrixNet) that was used to achieve 18th place.

How to use:
1. Modify data:
import AvazuDataTransform
path = 'dir' ###directory to where you store Avazu data
clTransform = AvazuDataTransform(path)
clTransform.get_Data_Transformed()

2. Get results for AdaGrad Logistic Regression and data for gbm:
import AvazuLR
path = 'dir' + 'LR_data\\' ###directory to where you store Avazu data
clLR = AvazuLR(path, alpha = 0.0003, n_passes = 6, poly = True, wTx=True, adagrad_start = 5)
clLR.main()

3. Get results for MatrixNet
###Load data into memory (requires 8-9 gb)
import MatrixNetData
data = MatrixNetData.MatrixNetData(path, 0.1613)
x_train, y_train, x_cv, y_cv = data.get_data()

###Run MatrixNet
import MatrixNet
mn = MatrixNet(x_train, y_train, x_cv, y_cv, fraction = 0.1, alpha = .3)
mn.train()
