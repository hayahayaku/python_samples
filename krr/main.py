import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump, load
import json

if __name__ == "__main__":
    data = pd.read_csv("wave.csv")
    x = np.array(data['x'])
    y = np.array(data['y'])
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    # print(pd.DataFrame({'x': x_tv.flatten(), 'y': y_tv}))
    pd.DataFrame({'x': x_tv, 'y': y_tv}).to_csv('train.csv', index=False)
    pd.DataFrame({'x': x_test, 'y': y_test}).to_csv('test.csv', index=False)
    lm_params = {'alpha' : np.linspace(0,0.2,20) , 'gamma' : np.linspace(0,5,30)}
    lm = KernelRidge(kernel="rbf")

    hypermodel = GridSearchCV(lm, lm_params, cv=5, scoring="neg_mean_squared_error")
    hypermodel.fit(x_tv.reshape(-1,1),y_tv)
    dump(hypermodel, 'model.joblib')

    y_pred_test = hypermodel.predict(x_test.reshape(-1,1))
    y_pred_tv = hypermodel.predict(x_tv.reshape(-1,1))

    scores = {
        'test_mae' : mean_absolute_error(y_test, y_pred_test),
        'test_mse' : mean_squared_error(y_test, y_pred_test),
        'test_r2' : r2_score(y_test, y_pred_test),
        'train_mae' : mean_absolute_error(y_tv, y_pred_tv),
        'train_mse' : mean_squared_error(y_tv, y_pred_tv),
        'train_r2' : r2_score(y_tv, y_pred_tv),
    }
    js = open('scores.json', 'w')
    js.write(json.dumps(scores))
    js.close()

    

    import matplotlib.pyplot as plt

    x_plot = np.linspace(-10,10,2000)
    y_plot = hypermodel.predict(x_plot.reshape(-1,1))

    def func(n):
        return np.exp(-(n/4)**2) * np.cos(4*n)

    plt.figure()
    plt.plot(x_plot, func(x_plot), label="f")
    plt.plot(x_plot, y_plot, label="predicted f")
    plt.scatter(x_tv.flatten(), y_tv, label="training data")
    plt.scatter(x_test.flatten(), y_test, label="test data")
    plt.legend()
    plt.title(f'MSE: {scores["test_mse"].__round__(3)}, MAE: {scores["test_mae"].__round__(3)}, R\u00b2: {scores["test_r2"].__round__(3)}')
    plt.savefig('plot.pdf')