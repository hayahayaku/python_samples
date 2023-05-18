import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve, cross_validate
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error, get_scorer_names
from joblib import dump

if __name__ == "__main__":
    data = pd.read_csv("nitride_compounds.csv", index_col=0)
    x = data.iloc[:,1:27].__array__()
    y = data['HSE Eg (eV)'].__array__()
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    lm_params = {'alpha' : [0.010526315789473684] , 'gamma' : [0.09090909090909091]}
    lm = KernelRidge(kernel="rbf")
    hypermodel = GridSearchCV(lm, lm_params, cv=5, scoring="neg_mean_squared_error")


    
    # Learning Curves
    train_sizes = np.linspace(0.1,0.9,9)
    _, _, test_scores = learning_curve(hypermodel, x_tv, y_tv, train_sizes=train_sizes, scoring='neg_mean_squared_error', cv=5)
    _, _, test_scores_r2 = learning_curve(hypermodel, x_tv, y_tv, train_sizes=train_sizes, scoring='r2', cv=5)
    mse = []
    r2 = []
    for each in range(0,9):
        mse.append(-test_scores[each].mean())
        r2.append(test_scores_r2[each].mean())



    # Best Model Training
    hypermodel.fit(x_tv,y_tv)
    dump(hypermodel, 'model.joblib')

    y_pred_test = hypermodel.predict(x_test)
    y_pred_tv = hypermodel.predict(x_tv)

    scores = {
        'test_mae' : mean_absolute_error(y_test, y_pred_test),
        'test_r2' : r2_score(y_test, y_pred_test),
    }



    # Plotting
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1,2,figsize=(13,6))
    
    axs0twin = axs[0].twinx()
    axs[0].set_title('Learning Curves')
    axs[0].set_xlabel('fraction of training data used')
    axs[0].plot(train_sizes, mse, color='darkorange')
    axs[0].set_ylabel('MSE', color='darkorange')
    axs[0].tick_params(axis='y', colors='darkorange')
    axs0twin.plot(train_sizes, r2, color='limegreen')
    axs0twin.set_ylabel('R\u00b2', color='limegreen')
    axs0twin.tick_params(axis='y', colors='limegreen')

    axs[1].plot([0,6], [0,6], color='lightblue')
    axs[1].scatter(y_tv, y_pred_tv, label="training data", color='darkorange')
    axs[1].scatter(y_test, y_pred_test, label="test data", color='limegreen')
    axs[1].set_xlabel('Calculated gap')
    axs[1].set_ylabel('Model gap')
    axs[1].legend()
    axs[1].set_title(f'Model R\u00b2: {scores["test_r2"].__round__(3)}, MAE: {scores["test_mae"].__round__(3)}')
    plt.savefig('plot.pdf')
    # plt.show()