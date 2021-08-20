# -*- coding: utf-8 -*-

import glob
import re
import os
import csv
import numpy as np
from configuration import GeoLocation_dict_continent, GeoLocation_dict_state

image_train_path = '/train/continent_images/'
trainfile_name='TrainImageNameList.csv'
valfile_name = 'ValImageNameList.csv'


#get file path of images
def read_imgname(path, p_dict):

    cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
    imgsName = []
    labels = []
    labelset = ()
    for idx, folder in enumerate(cate):
        label_name = re.sub(r'[^a-z0-9]+', ' ', folder.split('/')[-1].lower())
        idx = p_dict[label_name]
        print(idx)
        print(folder)

        for im in glob.glob(folder + '/*.jpg'):
            # print('reading the images:%s' % im)
            imgsName.append(im)
            labels.append(idx)

    print(len(imgsName))
    print(len(labels))
    return len(imgsName), imgsName, np.asarray(labels, np.int32)

count_img,imglist,label = read_imgname(image_train_path, GeoLocation_dict_continent)

# 打乱顺序
def shuffleImg(imglist,arr):
    assert len(imglist) == len(arr)
    imglist2=[]
    for i in range(0,len(imglist)):
        index=arr[i]
        imglist2.append(imglist[index])
    return imglist2

num_example = len(imglist)
arr = np.arange(num_example)
np.random.shuffle(arr)
data=shuffleImg(imglist,arr)
label = label[arr]

# 将所有数据分为训练集和验证集
ratio = 0.8
ratio2 = 1.0
s = np.int(num_example * ratio)
s2 = np.int(num_example * ratio2)
x_train = data[:s]
y_train = label[:s]
x_val = data[s:s2]
y_val = label[s:s2]

# 写进csv文件
def writeCSV(path,list1,list2):
    assert len(list1)==len(list2)
    csvFile = open(path, 'w', newline='')
    writer2 = csv.writer(csvFile)
    for i in range(0,len(list1)):
        writer2.writerow([list1[i], list2[i]])
    csvFile.close()

writeCSV(trainfile_name,x_train,y_train)
writeCSV(valfile_name,x_val,y_val)