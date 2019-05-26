#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22 下午1:02
# @Author  : FangZhou
# @Site    : 
# @File    : DeepL.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

ENPOCE = 1000
batch = 300

# 计算准确率 在测试的时候会用到
def compute_accuracy(v_xs, v_ys):  # 传入测试样本和对应的label
    global prediction #应为是个全局变量，在使用之前需要引用。
    # 得到预测
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # tf.argmax（～，0或1）返回行或者列中最大数的下表如下所示
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # tf.cast 此函数是类型转换函数
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# 计算weight
def weigth_variable(shape):
    # stddev : 正态分布的标准差
    initial = tf.truncated_normal(shape, stddev=0.1)  # 截断正态分布
    return tf.Variable(initial)


# 计算biases
def bias_varibale(shape):
    # stddev : 正态分布的标准差
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 计算卷积
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定placeholder
# 读取 x（x未知）个 784像素点的
xs = tf.placeholder(tf.float32, [None, 784]) / 255.
# 输入的标签
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)  # 用来处理过度拟合
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# 定义第一层
# 应用 32 个 5x5 过滤器（提取 5x5 像素的子区域），并应用 ReLU 激活函数
# 所以 '1' 应该是输入层
W_conv1 = weigth_variable([5, 5, 1, 32])
b_conv1 = weigth_variable([32])
# 卷积
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 28*28*32
# 池化
h_pool1 = max_pool_2x2(h_conv1)  # 14*14*32

# 定义第二层
# 卷积层 2：应用 64 个 5x5 过滤器，并应用 ReLU 激活函数
# '32' 代表上层的32个过滤器
W_conv2 = weigth_variable((5, 5, 32, 64))
b_conv2 = weigth_variable([64])
# 卷积
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 14*14*64
# 池化
h_pool2 = max_pool_2x2(h_conv2)  # 7*7*64

# 定义第三层全连接层
# 包含 1024 个神经元，其中丢弃正则化率为 0.4（任何指定元素在训练期间被丢弃的概率为 0.4）
# 如果需要更改图片尺寸需要注意这里 7 这个是通过卷积和池化算出来的。
W_fc1 = weigth_variable([7 * 7 * 64, 1024])
b_fc1 = bias_varibale([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 防止过度拟合
# 定义第四层全连接层
# 包含 10 个神经元，每个数字目标类别 (0–9) 对应一个神经元。
W_fc2 = weigth_variable([1024, 10])
b_fc2 = bias_varibale([10])

# 利用 softmax() 多项式逻辑回归， 代表每一个的概率
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 计算loss cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# 梯度下降优化
# 定义优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
sess = tf.Session()
# 初始化variable
init = tf.global_variables_initializer()
sess.run(init)
# 训练
for epoce in range(ENPOCE):
    batch_xs, batch_ys = mnist.train.next_batch(batch)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if epoce % 50 == 0:
        accuracy = compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000])
    print("epoch: %d  acc: %f" % (epoce + 1, accuracy))
