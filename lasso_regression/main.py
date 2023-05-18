import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import dump

if __name__ == "__main__":
    data = pd.read_csv("credit.csv", index_col=0)
    x = np.array([data['Rating'], data['Limit'], data['Cards'], data['Income']]).transpose()
    y = np.array(data['Balance'])
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.20)
    # lm = Lasso()
    lm_params = {'alpha' : np.arange(0,10001)}

    rating = []
    limit = []
    cards = []
    income = []
    r2 = []

    # lm_params = {'alpha' : np.arange(0,10001,100)}
    for alpha in lm_params['alpha']:
        model = Lasso(alpha=alpha)
        model.fit(x_tv, y_tv)
        # print(alpha, "coef", model.coef_)
        rating.append(model.coef_[0])
        limit.append(model.coef_[1])
        cards.append(model.coef_[2])
        income.append(model.coef_[3])
        y_pred = model.predict(x_test)
        # print(alpha, "r2", r2_score(y_pred, y_test))
        r2.append(r2_score(y_pred, y_test))
        # print("==========")
        if (alpha in np.arange(1,4)):
            dump(model, 'model_{}.joblib'.format(alpha))
        # print("done with",alpha)

    import matplotlib.pyplot as plt

    x_plot = lm_params['alpha']

    fig, axs = plt.subplots(2,1, sharex=True)

    axs[0].plot(x_plot, rating, label="Rating")
    axs[0].plot(x_plot, limit, label="Limit")
    axs[0].plot(x_plot, income, label="Income")
    axs[0].plot(x_plot, cards, label="Cards")
    axs[0].legend()
    axs[0].set_xscale('log')
    axs[0].set_ylabel('Value of the coefficient')

    axs[1].plot(x_plot, r2)
    axs[1].set_xlabel('alpha')
    axs[1].set_ylabel('R\u00b2')

    plt.savefig('plot.pdf')