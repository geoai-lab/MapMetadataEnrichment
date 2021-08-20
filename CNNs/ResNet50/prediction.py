from keras.models import load_model
from skimage import io, transform
import os
import glob
import numpy as np
from configuration import GeoLocation_dict_continent, model_continent_path


def read_image(path):
    imgs=[]
    label=[]
    for im in glob.glob(path + '/*.jpg'):
        img = io.imread(im)
        img = transform.resize(img, (224, 224, 3))

        imgs.append(img)
        label.append(os.path.basename(im))

    return len(imgs),np.asarray(imgs,np.float32),label


if __name__ == '__main__':
    test_image_path ='../../Test_data/Binary_maps/'
    img_rows, img_cols = 224, 224  # Resolution of inputs
    channel = 3
    num_classes = 8 
                              
    GeoLocation_dict_continent2 = {k: v for v, k in GeoLocation_dict_continent.items()}

    print("Begin to load ResNet model")
    model = load_model(model_continent_path)

    print("Begin to load test map images")

    all_accuracy = 0

    for region_name in GeoLocation_dict_continent2.keys():
        ## Read in all map images of each class
        print("Processing on the maps from category: " + region_name)
        index = int(GeoLocation_dict_continent2[region_name])
        imcount, data, label = read_image(test_image_path + region_name)
        data = data.reshape(imcount, img_rows, img_cols, channel)
        data = data[:, :, :, ::-1]

        # Subtract ImageNet mean pixel
        data[:, :, :, 0] -= 103.939
        data[:, :, :, 1] -= 116.779
        data[:, :, :, 2] -= 123.68

        # Model prediction on batch
        classification_result = model.predict(data)
        predicts = classification_result.argmax(axis=-1)

        # Calculate classification accuracy of this class
        accuracy = (predicts == index).sum()
        print("The classification accuracy of this class:")
        print(accuracy/len(predicts))

        all_accuracy += (accuracy/len(predicts))

        num_count=np.zeros(num_classes)
        for i in range(len(predicts)):
            print(classification_result[i][predicts[i]])
            num_count[predicts[i]]=num_count[predicts[i]]+1

        print("Error analysis of model classification:")
        print("Africa: %d "%(num_count[0]))
        print("Antarctica: %d " % (num_count[1]))
        print("Asia: %d " % (num_count[2]))
        print("Europe: %d " % (num_count[3]))
        print("Global: %d " % (num_count[4]))
        print("North America: %d " % (num_count[5]))
        print("Oceania: %d " % (num_count[6]))
        print("South America: %d " % (num_count[7]))

    print("The overall classification accuracy:")
    print(all_accuracy/num_classes)
