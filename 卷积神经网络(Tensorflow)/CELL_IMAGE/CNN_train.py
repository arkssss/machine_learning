#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/5/23 下午7:27
# @Author  : FangZhou
# @Site    : 
# @File    : CNN_train.py
# @Software: PyCharm
# %%

import os
import numpy as np
import tensorflow as tf
import CNN_readdata as rd
import CNN_model as model


# %%

N_CLASSES = 6
IMG_W = 78             # resize the image, if the input image is too large, training will be very slow.
IMG_H = 78
BATCH_SIZE = 30
CAPACITY = 2000
MAX_STEP = 20000        # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.00005  # with current parameters, it is suggested to use learning rate<0.0001

# dir to save the model
logs_dir = 'logs/try2/'

# %%
def run_training():
    # log dir
    logs_train_dir = logs_dir

    # read data
    test, train, validate, test_label, train_label, validate_label = rd.read_file()

    # get batch
    train_batch, train_label_batch = rd.get_batch(train,
                                                  train_label,
                                                  IMG_W,
                                                  IMG_H,
                                                  BATCH_SIZE,
                                                  CAPACITY)

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# %% Evaluate one image
# when training, comment the following codes.


from PIL import Image
import matplotlib.pyplot as plt


# def get_one_image(train):
#    '''Randomly pick one image from training data
#    Return: ndarray
#    '''
#    n = len(train)
#    ind = np.random.randint(0, n)
#    img_dir = train[ind]
#    # image = Image.open(img_dir)
#    # plt.imshow(image)
#    # # plt.show()
#    # image = image.resize([78, 78])
#    # image = np.array(image)
#    train_content = tf.read_file(img_dir)
#    # gray image
#    # use 0 means the gary image
#    train = tf.image.decode_png(train_content, channels=1)
#    # resize the image and crop or pad the image if nit fits
#    train = tf.image.resize_image_with_crop_or_pad(train, 78, 78)
#    # standardization the image
#    train = tf.image.per_image_standardization(train)
#    return train


# def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#
#    # you need to change the directories to yours.
#    # train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#
#    # test, train, validate, test_label, train_label, validate_label = rd.read_file()
#    # image = get_one_image(train)
#
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        # N_CLASSES = 2
#
#        # image = tf.cast(image_array, tf.float32)
#        # image = tf.image.per_image_standardization(image)
#        # gray image
#        # image = tf.reshape(image, [1, 78, 78, 1])
#        # logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#        #
#        # logit = tf.nn.softmax(logit)
#        #
#        # x = tf.placeholder(tf.float32, shape=[78, 78, 1])
#        #
#        # # you need to change the directories to yours.
#        # logs_train_dir = 'logs/'
#        #
#        # saver = tf.train.Saver()
#
#        # with tf.Session() as sess:
#        #
#        #     print("Reading checkpoints...")
#        #     ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#        #     if ckpt and ckpt.model_checkpoint_path:
#        #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#        #         saver.restore(sess, ckpt.model_checkpoint_path)
#        #         print('Loading success, global_step is %s' % global_step)
#        #     else:
#        #         print('No checkpoint file found')
#        #
#        #     prediction = sess.run(logit, feed_dict={x: image_array})
#        #     max_index = np.argmax(prediction)
#        #
#        #     print(prediction)
#            # if max_index==0:
#            #     print('This is a cat with possibility %.6f' %prediction[:, 0])
#            # else:
#            #     print('This is a dog with possibility %.6f' %prediction[:, 1])



# evaluate
def evaluate_all_image():
    '''
    Test all image against the saved models and parameters.
    Return global accuracy of test_image_set
    ##############################################
    ##Notice that test image must has label to compare the prediction and real
    ##############################################
    '''
    # you need to change the directories to yours.
    # test_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/test/'
    # N_CLASSES = 2
    print('-------------------------')
    test, train, validate, test_label, train_label, validate_label = rd.read_file()
    BATCH_SIZE = len(test)
    print('There are %d test images totally..' % BATCH_SIZE)
    print('-------------------------')
    test_batch, test_label_batch = rd.get_batch(test,
                                                test_label,
                                                IMG_W,
                                                IMG_H,
                                                BATCH_SIZE,
                                                CAPACITY)

    logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
    testloss = model.losses(logits, test_label_batch)
    testacc = model.evaluation(logits, test_label_batch)

    logs_train_dir = logs_dir
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found')
        print('-------------------------')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_loss, test_acc = sess.run([testloss, testacc])
        print('The model\'s loss is %.2f' % test_loss)
        correct = int(BATCH_SIZE * test_acc)
        print('Correct : %d' % correct)
        print('Wrong : %d' % (BATCH_SIZE - correct))
        print('The accuracy in test images are %.2f%%' % (test_acc * 100.0))
    coord.request_stop()
    coord.join(threads)
    sess.close()
if __name__ == '__main__':
    # run_training()
    # evaluate_one_image()
    evaluate_all_image()
