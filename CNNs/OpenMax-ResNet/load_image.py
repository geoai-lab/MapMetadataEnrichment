from skimage import io, transform
import numpy as np
import csv
import os


def readCSV(filepath):
    path = os.getcwd()
    parent_path = os.path.dirname(path)
    path = os.path.join(parent_path, filepath)
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    data = []
    labels = []
    for item in reader:
        data.append(item[0])
        labels.append(int(item[1]))
    csvFile.close()
    return data, np.asarray(labels, np.int32)


# read images from path
def read_img(imgs, img_rows=224, img_cols=224):
    imgs2 = []
    for item in imgs:
            #print('reading the images:%s' % item)
            img = io.imread(item)
            img = transform.resize(img, (img_rows, img_cols))
            imgs2.append(img)
    temp = len(imgs2)
    olddata = np.asarray(imgs2, np.float32)
    data = olddata.reshape(temp, img_rows, img_cols, 3)

    # Switch RGB to BGR order
    data = data[:, :, :, ::-1]

    # Subtract ImageNet mean pixel
    data[:, :, :, 0] -= 103.939
    data[:, :, :, 1] -= 116.779
    data[:, :, :, 2] -= 123.68

    return data


def sortImg(imglist, arr):
    imglist2 = []
    for i in range(0, len(arr)):
        index = arr[i]
        imglist2.append(imglist[index])
    return imglist2


# mini-batch
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False, ratio=1):
    assert len(inputs) == len(targets)
    # print("Need turn %f" % (len(inputs) / batch_size))
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
        imgName = sortImg(inputs, excerpt)
        imgData = read_img(imgName)
        yield imgData, targets[excerpt]

