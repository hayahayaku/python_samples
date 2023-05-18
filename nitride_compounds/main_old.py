import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split, KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_squared_error

if __name__ == "__main__":
    data = pd.read_csv("nitride_compounds.csv", index_col=0)

    x = data.iloc[:,1:27].__array__()
    y = data['HSE Eg (eV)'].__array__()
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    lm = KernelRidge()
    kf = KFold()
    print(x.shape, y.shape)
    # train_size_abs, train_scores, test_scores = learning_curve(lm, x, y, scoring="r2", train_sizes=np.linspace(0.1,0.9,9), cv=5)
    # print(train_scores, test_scores)

    for fraction in np.linspace(0.1,0.9,9):
        x_train, _, y_train, _ = train_test_split(x_tv, y_tv, train_size=fraction, random_state=1)
        model = KernelRidge()
        r2 = []
        mse = []
        # model = learning_curve(lm, x_train, y_train, scoring="neg_mean_squared_error")
        for train, test in kf.split(x_train,y_train):
            model.fit(x_train[train], y_train[train])
            y_pred = model.predict(x_train[test])
            r2this = r2_score(y_pred, y_train[test])
            r2.append(r2this)
        r2 = np.array(r2)
        r2 = r2.mean()
        print("==================================")
        print("for", fraction, ":", model.X_fit_.shape)
        print('r2', r2)
        # print('mse', mean_squared_error(y_test, y_pred))

    # model = KernelRidge()
    # model.fit(x_tv, y_tv)
    # y_pred = model.predict(x_test)
    # print("==================================")
    # print("for", "full", ":", model.X_fit_.shape)
    # print('r2', r2_score(y_test, y_pred))
    # print('mse', mean_squared_error(y_test, y_pred))
