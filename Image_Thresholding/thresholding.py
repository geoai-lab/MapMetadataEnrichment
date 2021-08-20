import os
import sys
from glob import glob
import numpy as np
from skimage import img_as_ubyte
from skimage import io
from skimage import color, transform
from skimage.filters import threshold_triangle, threshold_otsu, threshold_local, threshold_mean


rgb_image_path_continent = '/Test_data/Continents'
rgb_image_path_us = '/Test_data/States'
binary_image_path_continent = '/Test_data/Binary_maps/Continents'
binary_image_path_us = '/Test_data/Binary_maps/States'


def image_binary_global(img_filename, img_filename2):
    image = io.imread(img_filename)
    if image.ndim == 3:
        print("FROM RGB TO GRAY")
        image = color.rgb2gray(image)
    elif image.ndim == 4:
        print("FROM RGBA TO GRAY")
        image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    thresh = threshold_triangle(image)

    binary = img_as_ubyte(image > thresh)
    origin_row, origin_col = image.shape
    temp1 = max(origin_col, origin_row)
    temp2 = min(origin_col, origin_row)

    if (temp1/temp2) > 1.5:
        print("not appropriate to be resized")
        temp1 = int(1.1*temp1)
        image2 = np.full((temp1, temp1), 0, dtype='ubyte')

        image2[int(temp1/2 - origin_row/2): int(temp1/2 - origin_row/2) +origin_row, \
               int(temp1/2 - origin_col/2): int(temp1/2 - origin_col/2)+origin_col] = binary
        image2 = transform.resize(image2, (300, 300), anti_aliasing=True)
    else:
        image2 = transform.resize(binary, (300, 300), anti_aliasing=True)

    io.imsave(img_filename2, img_as_ubyte(image2))


def workflow(path1, path2):
    origin_cate = [path1 + '/' + x for x in os.listdir(path1) if os.path.isdir(path1 + '/' + x)]
    new_cate = [path2 + '/' + x for x in os.listdir(path1) if os.path.isdir(path1 + '/' + x)]

    for idx, folder in enumerate(origin_cate):
        name_id = 1
        img_new_path = new_cate[idx]
        is_exists = os.path.exists(img_new_path)
        if not is_exists:
            os.makedirs(img_new_path)

            files = []
            for ext in ('*.gif', '*.png', '*.jpg'):
                files.extend(glob(os.path.join(folder, ext)))

            for img in files:
                print(img)
                name_id += 1
                img_name = str(name_id) + ".jpg"
                img_save_path = os.path.join(img_new_path, img_name)
                image_binary_global(img_filename=img, img_filename2=img_save_path, option="triangle")


workflow(path1=rgb_image_path_continent, path2=binary_image_path)
workflow(path1=rgb_image_path_us, path2=binary_image_path)