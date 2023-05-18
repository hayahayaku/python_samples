import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

model = load('model.joblib')
# print(model)

# d_train = pd.read_csv("train.csv", header=None)
d_train = pd.read_csv("train.csv")
x_train = np.array(d_train.iloc[:,0])
y_train = np.array(d_train.iloc[:,1])

# d_test = pd.read_csv("test.csv", header=None)
d_test = pd.read_csv("test.csv")
x_test = np.array(d_test.iloc[:,0])
y_test = np.array(d_test.iloc[:,1])

y_pred_test = model.predict(x_test.reshape(-1,1))
y_pred_tv = model.predict(x_train.reshape(-1,1))

print(x_test.shape, x_train.shape)

print('test_mae',  mean_absolute_error(y_test, y_pred_test))
print('test_mse', mean_squared_error(y_test, y_pred_test))
print('test_r2', r2_score(y_test, y_pred_test))
print('train_mae',  mean_absolute_error(y_train, y_pred_tv))
print('train_mse', mean_squared_error(y_train, y_pred_tv))
print('train_r2', r2_score(y_train, y_pred_tv))