# -*- coding: utf-8 -*-
from skimage import io, transform, color
import csv
import tensorflow as tf
import numpy as np
import time
import os
import sys
from configuration import model_save_path

import alexnet

testfile_name='ValImageNameList.csv'
trainfile_name='TrainImageNameList.csv'


# resize images into 227*227
r = 227
c = 227
dim = 3
# Numbers of classes in the prediction
num_classes = 8
# Not to load the initial weights from AlexNet
not_load_weights_layers = ['fc8']
# Layers fine tuned in the learning process
train_layers=['fc8', 'fc7','fc6','conv1','conv2']
# drop out rate
dropout_rate = 0.5

# read training and testing dataset path from the metadata file
def readCSV(filepath):
    path = os.getcwd()
    parent_path = os.path.dirname(path)
    path = parent_path + filepath
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    data = []
    labels=[]
    for item in reader:
        data.append(item[0])
        labels.append(int(item[1]))
    csvFile.close()
    return data,np.asarray(labels, np.int32)

# read images from path
def read_img(list):
    imgs = []
    for item in list:
            #print('reading the images:%s' % item)
            img = io.imread(item)
            img = transform.resize(img, (r, c))
            img = color.gray2rgb(img)
            imgs.append(img)
    temp=len(imgs)
    olddata = np.asarray(imgs, np.float32)
    data = olddata.reshape(temp, r, c, dim)
    return data

def sortImg(imglist,arr):
    imglist2=[]
    for i in range(0,len(arr)):
        index=arr[i]
        imglist2.append(imglist[index])
    return imglist2

# mini-batch
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False, ratio = 5):
    assert len(inputs) == len(targets)
    print("Need turn %f" % (len(inputs) / (ratio*batch_size)))
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    else:
        indices = np.arange(len(inputs))
    for start_idx in range(0, len(inputs) - (ratio*batch_size) + 1, ratio*batch_size):
        # if shuffle:
        excerpt = indices[start_idx:start_idx + batch_size]
        # print("the start_idx:%d" % start_idx)
        # else:
        #     excerpt = indices(start_idx, start_idx + batch_size)
        #     print("the start_idx:%d" % start_idx)
        imgName = sortImg(inputs,excerpt)
        imgData=read_img(imgName)
        yield imgData, targets[excerpt]

# -----------------Construction of AlexNet Model----------------------
# Define the structure
# TF placeholder for graph input and output

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, r, c, dim],name='x')
    y_ = tf.placeholder(tf.int32, [None, ],name='y_')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Initialize model
    model = alexnet.AlexNet(x, keep_prob, num_classes, not_load_weights_layers)

    # Link variable to model output
    score = model.fc8
    # ---------------------------End of architecture---------------------------
    # Multiply logits by 1 -> logits_eval，define the name that are used in future to get the variable

    b = tf.constant(value=1,dtype=tf.float32)
    logits_eval = tf.multiply(score,b,name='logits_eval')

    gb_step = tf.Variable(0,trainable=False)
    p_lr=tf.train.exponential_decay(0.001,gb_step,1000,0.95,staircase=True)

    # p_var_list  = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=score)
    train_op = tf.train.AdamOptimizer(learning_rate=p_lr).minimize(loss,global_step=gb_step)
    correct_prediction = tf.equal(tf.cast(tf.argmax(score, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Save everything in the fine tuned AlexNet
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

# Training and validating process
print("read image name list--start")
x_train,y_train = readCSV(filepath=trainfile_name)
x_val,y_val = readCSV(filepath=testfile_name)
print("read image name list--end")

with tf.Session(graph=graph) as sess:
    n_epoch = 50
    batch_size = 64
    batch_size_val = 64
    sess.run(init)
    model.load_initial_weights(sess)
    
    for epoch in range(n_epoch):
        start_time = time.time()
        print("start the %d training" % epoch)
        # training
        train_loss, train_acc, n_batch,LR = 0, 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, ratio=4, shuffle=True):
            _, err, ac,learningRate = sess.run([train_op, loss, acc,p_lr], feed_dict={x: x_train_a, y_: y_train_a, keep_prob: dropout_rate})
            train_loss += err
            train_acc += ac
            n_batch += 1
            LR=learningRate
        print("  train loss: %f" % (train_loss / n_batch))
        print("  train acc: %f" % (train_acc / n_batch))
        print("  learning rate: %f" % LR)

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size_val, ratio=2, shuffle=True):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a, keep_prob: dropout_rate})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("  validation loss: %f" % (val_loss / n_batch))
        print("  validation acc: %f" % (val_acc / n_batch))
        temp = val_acc/n_batch
        if temp > 0.98 :
            break
    saver.save(sess, model_save_path)