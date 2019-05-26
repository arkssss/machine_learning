#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/18 下午5:25
# @Author  : FangZhou
# @Site    : 
# @File    : plot_member.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = pd.read_table("res/tsp100_bestMember.txt")
city = pd.read_table("dataset/tsp100.txt")

city.columns = pd.Index(['x'])
city = city['x'].str.split(' ', expand=True)
city.columns = ['x', 'y', 'type']

city['x'] = city['x'].apply(lambda x: int(x))
city['y'] = city['y'].apply(lambda x: int(x))

path = path['path'].str.split(' ', expand=True)
path.drop_duplicates(subset=None, keep='first', inplace=True)
path.drop(100, axis=1, inplace=True)
path.index = range(len(path))

# print(path)

plt.figure(figsize=(8, 6), dpi=80)
plt.ion()

for i in range(path.shape[0]):
    # plt.cla()

    last_path = path.loc[i]
    # print(city)
    # print(city.loc[0])
    x_pos = []
    y_pos = []
    for i in range(0, 100):
        path_loc = int(last_path[i])
        the_city = city.loc[path_loc]
        x_pos.append(the_city['x'])
        y_pos.append(the_city['y'])

    New_path = pd.DataFrame(columns=list('xy'))
    New_path['x'] = x_pos
    New_path['y'] = y_pos
    # plt
    New_path.plot(x='x', y='y', color='m', marker='o')
    plt.savefig('plt/'+str(i)+".svg")
    plt.plot(x=city['x'], y=city['y'])
    plt.pause(0.1)


# plt.show()

# last_path = path.loc[599]
# # print(city)
# # print(city.loc[0])
# x_pos = []
# y_pos = []
# for i in range(0, 100):
#     path_loc = int(last_path[i])
#     the_city = city.loc[path_loc]
#     x_pos.append(the_city['x'])
#     y_pos.append(the_city['y'])
#
# New_path = pd.DataFrame(columns=list('xy'))
# New_path['x'] = x_pos
# New_path['y'] = y_pos
# print(New_path)

# plt

# New_path.plot(x='x', y='y', color='m', marker='o')
# plt.plot(x=city['x'], y=city['y'])
# plt.show()







