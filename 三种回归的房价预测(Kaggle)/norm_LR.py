#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/10 下午9:50
# @Author  : FangZhou
# @Site    : 
# @File    : norm_LR.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

train = pd.read_csv("data/train_after_processing.csv")
test = pd.read_csv("data/test_after_processing.csv")

drop_feats = ['OverallQual', 'GrLivArea']
train.drop(drop_feats, axis=1, inplace=True)
test.drop(drop_feats, axis=1, inplace=True)

init_test = pd.read_csv('data/test.csv')
if __name__ == '__main__':
    # # build a model
    # # spilt target and feats
    feats_label = train.columns
    feats = train.loc[:, feats_label[:-1]]
    target = train.loc[:, 'SalePrice']
    # look for the best parameter
    norm = linear_model.LinearRegression()
    norm.fit(feats, target)

    # use the Minimum value
    norm.fit(feats, target)
    # 这两行
    score = cross_val_score(norm, feats, target, cv=5)
    print("the final score with normal linear model is :" + str(np.around(np.mean(score) * 100, decimals=2)) + "%")

    # predict_price = norm.predict(test)
    # predict_price = pd.DataFrame(
    #     predict_price,
    #     columns=['SalePrice']
    # )
    #
    # predict_price = predict_price.apply(lambda x: np.exp(x) - 1)
    # predict_price['Id'] = init_test['Id']
    #
    # predict_price.to_csv('res/norm_predict.csv', index=False)