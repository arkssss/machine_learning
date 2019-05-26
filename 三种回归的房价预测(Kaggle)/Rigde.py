#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/8 下午1:44
# @Author  : FangZhou
# @Site    : 
# @File    : Rigde.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

train = pd.read_csv("data/train_after_processing.csv")
test = pd.read_csv("data/test_after_processing.csv")


init_test = pd.read_csv('data/test.csv')

if __name__ == '__main__':

    # alphas = [0.001, 0.005, 0.01, 0.1, 0.2, 0.5, 0.8, 2.0, 5.0, 8.0, 10.0, 11.0, 12.0, 30.0, 50.0]
    # define the range of our alpha and find the best alpha
    alphas = np.arange(0.1, 50.0, 0.1)
    # # build a model
    errors_rigde = []
    # # spilt target and feats
    feats_label = train.columns
    feats = train.loc[:, feats_label[:-1]]
    target = train.loc[:, 'SalePrice']
    # look for the best parameter
    for alpha in alphas:
        # reg = linear_model.RidgeCV(alphas=alphas)
        Ridge = linear_model.Ridge(alpha=alpha)
        # print(feats_label)
        Ridge.fit(feats, target)
        # errors.append(rmse(reg.predict(feats), target))
        # use the cross_val_score to represent the fittness of each model
        errors_rigde.append(np.sqrt(-cross_val_score(Ridge, feats, target, scoring="neg_mean_squared_error", cv=5)).mean())
        # errors_lasso.append(np.sqrt(-cross_val_score(Lasso, feats, target, scoring="neg_mean_squared_error", cv=5)).mean())

    errors_rigde = pd.Series(errors_rigde, index=alphas)
    print(errors_rigde.describe())
    print("the Rigde best alpha is :" + str(errors_rigde.idxmin()))

    # al_error.to_csv('data/rigde_error.csv')
    plt.xlabel('alpha')
    plt.ylabel('rmse')
    errors_rigde.plot()
    plt.show()


    # use the Minimum value
    reg = linear_model.Ridge(alpha=7.75)
    # reg.fit(feats, target)
    score = cross_val_score(reg, feats, target, cv=5)
    print("the final score with normal linear model is :" + str(np.around(np.mean(score) * 100, decimals=2)) + "%")

    print(np.sqrt(-cross_val_score(reg, feats, target, scoring="neg_mean_squared_error", cv=5)).mean())
    # predict the test
    predict_price = reg.predict(test)
    predict_price = pd.DataFrame(
        predict_price,
        columns=['SalePrice']
    )

    predict_price = predict_price.apply(lambda x: np.exp(x) - 1)
    predict_price['Id'] = init_test['Id']

    predict_price.to_csv('Rigde_predict.csv', index=False)

