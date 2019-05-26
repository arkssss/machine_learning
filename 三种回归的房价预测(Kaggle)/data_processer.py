#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/6 下午12:02
# @Author  : FangZhou
# @Site    : 
# @File    : data_processer.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


Step1 = True
Step2 = True
Step3 = True
Step4 = True

train_url = 'data/train.csv'
test_url = 'data/test.csv'
Stew_threshold = 0.75

if __name__ == '__main__':
    """
    all step to preprocessing data
    1.  drop the Outliers 

    2.  Replace the missing value
        Create Dummy variables for the categorical features
        Replace the numeric missing values (NaN's) with the mean of their respective columns
    
    3.  Stew the data (SalePrice & others)
        First I'll transform the skewed numeric features by taking log(feature + 1) - this will make the features more normal
        
    4.  Norm the data
    """
    # read the data set
    train_data = pd.read_csv(train_url)
    test_data = pd.read_csv(test_url)

    # save col of Id
    train_ID = train_data['Id']
    test_ID = test_data['Id']

    # drop col of Id
    # axis = 1 means drop the col (0 means the row)
    # inplace means the new_data replace the old one
    train_data.drop("Id", axis=1, inplace=True)
    test_data.drop("Id", axis=1, inplace=True)

    # select all the numeric feats
    numeric_feats = train_data.dtypes[train_data.dtypes != 'object'].index

    # select all the none-numeric feats
    object_feats = train_data.dtypes[train_data.dtypes == 'object'].index


    # Step 1 . ------------------------------------------------> Drop the Outliers
    if Step1:
        # draws all the numeric feats to select the Outliers Manually
        # for feats in numeric_feats:
        #     # Outliers
        #     fig, ax = plt.subplots()
        #     ax.scatter(x=train_data[feats], y=train_data['SalePrice'])
        #     plt.ylabel('SalePrice', fontsize=13)
        #     plt.xlabel(feats, fontsize=13)
        #     # plt.show()
        #     plt.savefig('plt/Outliers/before/'+feats+'.svg')

        # draw all the figure with outliers
        # times = 1
        #
        # for feats in ["1stFlrSF", "BsmtFinSF1", 'GrLivArea', 'LotFrontage', 'TotalBsmtSF']:
        #     plt.subplot(2, 3, times)
        #     plt.scatter(x=train_data[feats], y=train_data['SalePrice'] / 1000)
        #     plt.ylabel('SalePrice (k)', fontsize=12)
        #     plt.xlabel(feats, fontsize=12)
        #     times += 1
        # plt.show()




        # find the Outliers and remove all the Outliers
        # actually there are just two points
        Outliers = train_data[(train_data['1stFlrSF'] > 4000)].index.tolist()
        Outliers.extend(train_data[train_data['BsmtFinSF1'] > 5000].index.tolist())
        Outliers.extend(train_data[(train_data['GrLivArea'] > 4000) & (train_data['SalePrice'] < 200000)].index.tolist())
        Outliers.extend(train_data[train_data['LotFrontage'] > 300].index.tolist())
        Outliers.extend(train_data[train_data['TotalBsmtSF'] > 6000].index.tolist())

        Outliers = list(set(Outliers))

        # drop
        train_data.drop(pd.Series(Outliers), inplace=True)

        times = 1

        for feats in ["1stFlrSF", "BsmtFinSF1", 'GrLivArea', 'LotFrontage', 'TotalBsmtSF']:
            plt.subplot(2, 3, times)
            plt.scatter(x=train_data[feats], y=train_data['SalePrice'] / 1000)
            plt.ylabel('SalePrice (k)', fontsize=12)
            plt.xlabel(feats, fontsize=12)
            times += 1
        plt.show()

        # fill the NA to None
        for object_feat in object_feats:
            train_data[object_feat].fillna('None', inplace=True)
            print(train_data[object_feat].value_counts())

        # draws all the feats after being dropped Outliers
        # for feats in numeric_feats:
        #     # Outliers
        #     fig, ax = plt.subplots()
        #     ax.scatter(x=all_data[feats], y=all_data['SalePrice'])
        #     plt.ylabel('SalePrice', fontsize=13)
        #     plt.xlabel(feats, fontsize=13)
        #     # plt.show()
        #     plt.savefig('plt/Outliers/after/'+feats+'.svg')

        # combine two DataFrame
        all_data = pd.concat(
            (train_data.loc[:, 'MSSubClass':'SaleCondition'], test_data.loc[:, 'MSSubClass':'SaleCondition']))

        numeric_feats = numeric_feats.drop("SalePrice")

        # des = all_data[numeric_feats].describe()
        # des = des.drop(['count', '25%', '50%', '75%', 'max'])
        # des = des.T
        # des.plot(kind='bar')
        # plt.show()

        # times = 1
        # for object_feat in object_feats[:25]:
        #     plt.subplot(5, 5, times)
        #     ax = sns.boxplot(x=object_feat, y="SalePrice", data=train_data)
        #     times += 1
        # plt.show()
        # times = 1
        # for object_feat in object_feats[25:]:
        #     plt.subplot(5, 5, times)
        #     ax = sns.boxplot(x=object_feat, y="SalePrice", data=train_data)
        #     times += 1
        # plt.show()

        times = 1
        for numeric_feat in numeric_feats[:5]:
            plt.subplot(2, 3, times)
            plt.hist(pd.DataFrame({numeric_feat: all_data[numeric_feat]}))
            times += 1
        plt.show()

    # 2.Step Two ----------------------------------------------> Deal the missing data
    if Step2:
        # see the init miss rate
        plt.rcParams['figure.figsize'] = (12.0, 12.0)
        missing_rate = all_data.isnull().sum() / len(all_data) * 100
        missing_rate.sort_values(inplace=True)
        missing_rate.plot.barh(stacked=True, width=1.0)
        plt.ylabel('Features', fontsize=15)
        plt.xlabel('Percent of missing values', fontsize=15)
        plt.title('Percent of missing values by features', fontsize=15)
        plt.savefig('plt/Missing/init.svg')

        # if it is numeric_feats then fill with the mean
        # drop the col of SalePrice
        numeric_feats = numeric_feats.drop('SalePrice')
        all_data[numeric_feats] = all_data[numeric_feats].fillna(all_data.mean())

        # get_dummies of all the feats to deal the object feats
        all_data = pd.get_dummies(all_data)

        # Now the missing rate is 0

        # corrmat = train_data.corr()
        # print(corrmat[abs(corrmat['SalePrice']) > 0.7].index)
        # plt.subplot(figsize=[30,15])
        # sns.heatmap(corrmat)
        # plt.show()

    # 3.Step Three --------------------------------------------> Stew the data
    if Step3:
        # rcParams function use to set the config of the figure
        # igure.figsize means that it need to be a 12 * 6 figure
        plt.rcParams['figure.figsize'] = (12.0, 6.0)
        prices = pd.DataFrame({"price": train_data['SalePrice'], "log(price + 1)": np.log1p(train_data['SalePrice'])})
        prices.hist()

        # if not show ,the img will not display
        plt.show()
        plt.savefig('plt/Stew/SalePrice.svg')

        # Stew the SalePrice
        train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
        # prices = pd.DataFrame(    {"price": all_data['SalePrice']})
        # prices.hist()
        # plt.show()
        # print(train_data.columns)
        # print(test_data.columns)

        # Stew the other value

        # compute the stewness of all the feats
        # x.dropna() can filler the NA number
        Skewness = train_data[numeric_feats].apply(lambda x: x.skew())
        Skewness_feats = Skewness[Skewness > Stew_threshold].index
        Skewness.sort_values(inplace=True)
        Skewness.plot.barh(stacked=True)
        plt.savefig('plt/Stew/AllFeats.svg')
        # stew it
        all_data[Skewness_feats] = np.log1p(all_data[Skewness_feats])

    if Step4:
        # Norm
        all_data['LotFrontage'] = all_data['LotFrontage'].apply(
            lambda x: (x - all_data['LotFrontage'].min()) / (all_data['LotFrontage'].max() - all_data['LotFrontage'].min()))
        all_data['YearBuilt'] = all_data['YearBuilt'].apply(
            lambda x: (x - all_data['YearBuilt'].min()) / (all_data['YearBuilt'].max() - all_data['YearBuilt'].min()))
        all_data['YearRemodAdd'] = all_data['YearRemodAdd'].apply(
            lambda x: (x - all_data['YearRemodAdd'].min()) / (all_data['YearRemodAdd'].max() - all_data['YearRemodAdd'].min()))

    # save_train = all_data[0:train_data.shape[0]]
    # save_test = all_data[train_data.shape[0]:]

    # save_train = pd.concat([save_train, train_data['SalePrice']])
    # save_train.loc[:, 'SalePrice'] = train_data['SalePrice'].values

    # Save to csv
    # save_train.insert(save_train.shape[1], 'SalePrice', train_data['SalePrice'])
    # save_train.to_csv('data/train_after_processing.csv', index=False)
    # save_test.to_csv('data/test_after_processing.csv', index=False)
























