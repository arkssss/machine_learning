#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/22 下午1:43
# @Author  : FangZhou
# @Site    : 
# @File    : CNN_Cell_Image.py
# @Software: PyCharm
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

test_dir = "CellImage_data/test/"
training_dir = "CellImage_data/training/"
validation_dir = "CellImage_data/validation/"
label = "gt_training.csv"

Image_class = {
    'Speckled': 0,
    'Centromere': 1,
    'Nucleolar': 2,
    'Homogeneous': 3,
    'NuMem': 4,
    'Golgi': 5
}

# image_width = 200
# image_height = 200

# 每一个批次的数据数量
# batch_size = 2

# the MAX item in queue
# capacity = 200

def read_file():
    """
    read the cell image

    :return:
    """

    train = []
    train_label = []


    test = []
    test_label = []


    validate = []
    validate_label =[]

    # label
    data_label = pd.read_csv(label)

    # print(data_label['Image class'].value_counts())
    data_label['Image class'] = data_label['Image class'].apply(lambda x: Image_class[x])

    # print(data_label)
    # test
    for file in os.listdir(test_dir):
        name = file.split('.')
        test_id = name[0]
        # append the image
        test.append(test_dir + file)
        # append the label
        # test_label.append()
        test_label.append(data_label[data_label['Image ID'] == int(test_id)]['Image class'].values[0])
    # train
    for file in os.listdir(training_dir):
        name = file.split('.')
        train_id = name[0]
        # append the image
        train.append(training_dir + file)
        # append the label
        train_label.append(data_label[data_label['Image ID'] == int(train_id)]['Image class'].values[0])
    # validate
    for file in os.listdir(validation_dir):
        name = file.split('.')
        validate_id = name[0]
        validate.append(training_dir + file)
        validate_label.append(data_label[data_label['Image ID'] == int(validate_id)]['Image class'].values[0])

    print('Load %d test data, %d train data, %d validate data successful'%(len(test), len(train), len(validate)))
    return test, train, validate, test_label, train_label, validate_label


def get_batch(train, train_label, image_width, image_height, batch_size, capacity):
    """

    :return:
    """
    # make list become the tensor
    train = tf.cast(train, tf.string)
    # label type must be int
    train_label = tf.cast(train_label, tf.int32)

    # make an input queue
    input_data = tf.train.slice_input_producer([train, train_label])

    # the label
    train_label = input_data[1]
    train_contents = tf.read_file(input_data[0])
    # gray image
    # use 0 means the gary image
    train = tf.image.decode_png(train_contents, channels=1)
    ######################################
    # data argumentation should go to here
    # train is image , train label is the label
    # do some argumentation
    # train_flap = tf.image.random_flip_left_right(train)
    # sess = tf.Session()
    # train_flap_val = sess.run(train_flap)
    # print(train_flap_val.shape)


    ######################################


    # resize the image and crop or pad the image if nit fits
    train = tf.image.resize_image_with_crop_or_pad(train, image_width, image_height)

    # standardization the image
    train = tf.image.per_image_standardization(train)
    # get batch
    train_batch, label_batch = tf.train.batch([train, train_label],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity
                                              )
    label_batch = tf.reshape(label_batch, [batch_size])
    train_batch = tf.cast(train_batch, tf.float32)

    return train_batch, label_batch


if __name__ == '__main__':
    #  read_file
    test, train, validate, test_label, train_label, validate_label = read_file()

    #  get the batch
    train_batch, label_batch = get_batch(train, train_label)

    # with tf.Session() as sess:
    #     i = 0
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     try:
    #         while not coord.should_stop() and i < 1:
    #             img, label = sess.run([train_batch, label_batch])
    #             for j in range(batch_size):
    #                 print(label[j])
    #                 plt.imshow(img[j, :, :, :])
    #                 plt.show()
    #             i += 1
    #     except tf.errors.OutOfRangeError:
    #         print("dnoe")
    #     finally:
    #         coord.request_stop()
    #     coord.join(threads)












