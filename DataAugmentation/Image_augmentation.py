# -*- coding: utf-8 -*-

import os
import cv2
import copy
import glob
import numpy as np
import random
from skimage.util import random_noise, img_as_ubyte

shf_orgImagePath='../Training_data/base_image/Continents'
shf_trainImagebasePath='../Training_data/image_set/Continents'

#shf_orgImagePath='../Training_data/single_aug/input'
#shf_trainImagebasePath='../Training_data/single_aug/output'

#shf_orgImagePath='../Training_data/base_image/States'
#shf_trainImagebasePath='../Training_data/image_set/States'


scaling_ratios = [0.8, 0.85, 0.90, 0.95, 1, 1.05, 1.1, 1.15, 1.2]
resize_ratios = [0.5, 0.75, 1, 1.25, 1.5]
rotation_degrees1 = [-30, -20, -10, 0, 10, 20, 30]
rotation_degrees2 = [0]

# read image as array using OpenCV
def readImg(imPath):
	img = cv2.imread(imPath, cv2.IMREAD_GRAYSCALE)
	ret, thresh2 = cv2.threshold(img,250, 255, cv2.THRESH_BINARY_INV)
	return thresh2


# get the bbox of geographic area
def imgBBoxCut(img):
	x, y, width, height = cv2.boundingRect(img)
	img=img[y:y+height,x:x+width]
	return img


def createName(path,index):
	tempList = [path, '/', str(index), '.jpg']
	imgPath = ''.join(tempList)
	return imgPath


def scale_in(img, ratio):
	rows = img.shape[0]
	cols = img.shape[1]
	new_img = np.full([rows, cols], 0, dtype=np.uint8)
	img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
	new_r = img.shape[0]
	new_c = img.shape[1]
	startR = int((rows - new_r) / 2)
	startC = int((cols - new_c) / 2)
	new_img [startR:startR + new_r, startC:startC + new_c]= img

	return new_img


def scale_out(img, ratio):
	rows = img.shape[0]
	cols = img.shape[1]
	new_img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
	new_r = img.shape[0]
	new_c = img.shape[1]
	startR = int((new_r - rows) / 2)
	startC = int((new_c - cols) / 2)
	img = new_img [startR:startR + rows, startC:startC + cols]

	return img

def rotate_image_uncrop(mat, angle):
	"""
	Rotates an image (angle in degrees) and expands image to avoid cropping
	"""
	if angle == 0:
		return mat

	else:
		height, width = mat.shape[:2] # image shape has 3 dimensions
		image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

		rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

		# rotation calculates the cos and sin, taking absolutes of those.
		abs_cos = abs(rotation_mat[0,0])
		abs_sin = abs(rotation_mat[0,1])

		# find the new width and height bounds
		bound_w = int(height * abs_sin + width * abs_cos)
		bound_h = int(height * abs_cos + width * abs_sin)

		# subtract old image center (bringing image back to origo) and adding the new image center coordinates
		rotation_mat[0, 2] += bound_w/2 - image_center[0]
		rotation_mat[1, 2] += bound_h/2 - image_center[1]

		# rotate image with the new bounds and translated rotation matrix
		rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
		return rotated_mat


def img_augment(img, path, nameid):
	i = nameid

	for temp1 in scaling_ratios:
		for temp2 in resize_ratios:
			for temp3 in rotation_degrees1:

				## scaling operation
				scaling_img = img
				if temp1 < 1:
					scaling_img = scale_in(img, temp1)
				elif temp1 > 1:
					scaling_img = scale_out(img, temp1)

				## resize operation
				resize_img = cv2.resize(scaling_img, (int(scaling_img.shape[1] * temp2), scaling_img.shape[0]), interpolation=cv2.INTER_LINEAR)
				resize_img = imgPadding(resize_img)

				## rotate operation
				rotation_img = rotate_image_uncrop(resize_img, temp3)

				i = i + 2
				imgBBOXSave(rotation_img, path, i)

				# noise operation
				i = i + 2
				noise_img =  copy.deepcopy(rotation_img)
				imgBBOXSave(noise_img, path, i, if_noise=True)

				## blur operation

				i = i + 2
				kernel_size = random.choice([3,5,7,9,11])
				res_blur = cv2.GaussianBlur(rotation_img, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
				imgBBOXSave(res_blur, path, i)

	return i


# this is a test function to see the effective of individual augmentation strategies
def single_augment_test(img, path, nameid):
	
	i=nameid
	# save projection
	proj_img = imgPadding(img)
	imgBBOXSave(proj_img, path, i)
	print("map projection is: "+str(i))
	
	
	## scaling operation
	i=i+2
	scaling_img = scale_in(img, 0.8)
	scaling_img = imgPadding(scaling_img)
	imgBBOXSave(scaling_img, path, i)
	print("scaling is: "+str(i))
	
	
	## resize operation
	i=i+2
	resize_img = cv2.resize(img, (int(img.shape[1] * 1.5), img.shape[0]), interpolation=cv2.INTER_LINEAR)
	resize_img = imgPadding(resize_img)
	imgBBOXSave(resize_img, path, i)
	print("resizing is: "+str(i))
	
	
	## rotate operation
	i=i+2
	rotation_img = rotate_image_uncrop(img, 30)
	rotation_img = imgPadding(rotation_img)
	imgBBOXSave(rotation_img, path, i)
	print("rotating is: "+str(i))
	
	
	# noise operation
	i = i + 2
	noise_img =  copy.deepcopy(img)
	noise_img = imgPadding(noise_img)
	imgBBOXSave(noise_img, path, i, if_noise=True)
	print("noise is: "+str(i))


	## blur operation
	i = i + 2
	kernel_size = random.choice([3,5,7,9,11])
	res_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)
	res_blur = imgPadding(res_blur)
	imgBBOXSave(res_blur, path, i)
	print("blur is: "+str(i))
				

	return i


# Binarization convert
def convColor(pimg):
	img = copy.deepcopy(pimg)
	img[img <= 100] = 190
	img[img >= 200] = 0
	img[img == 190] = 255
	return img


def imgPadding(img):
	length = round(1.1 * max(img.shape[0], img.shape[1]))
	emptyImage = np.full([length, length], 0, dtype=np.uint8)
	rows = img.shape[0]
	cols = img.shape[1]
	startR = int((length - rows) / 2)
	startC = int((length - cols) / 2)
	emptyImage[startR:startR + rows, startC:startC + cols] = img[0:rows, 0:cols]
	return emptyImage


def imgFill(img,height,width):
	length = round(1.2 * max(height, width))
	emptyImage = np.full([length, length,3], 0, dtype=np.uint8)
	rows = img.shape[0]
	cols = img.shape[1]
	startR = int((length - rows) / 2)
	startC = int((length - cols) / 2)
	emptyImage[startR:startR + rows - 1, startC:startC + cols - 1, 0] = 200*(img[0:rows - 1, 0:cols - 1]/255)
	emptyImage[startR:startR + rows - 1, startC:startC + cols - 1, 1] = 232*(img[0:rows - 1, 0:cols - 1]/255)
	emptyImage[startR:startR + rows - 1, startC:startC + cols - 1, 2] = 200*(img[0:rows - 1, 0:cols - 1]/255)
	return emptyImage


def imgBBOXSave(img,path,flag, if_noise=False):
	img_path = createName(path, flag)
	img_path2 = createName(path, flag+1)
	
	if if_noise is False:
		img_rgb = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
		cv2.imwrite(img_path, img_rgb)
		cv2.imwrite(img_path2, convColor(img_rgb))

	else:
		temp = img_as_ubyte(random_noise(convColor(img).astype(float), mode='gaussian'))
		img_rgb = cv2.cvtColor(temp,cv2.COLOR_GRAY2RGB)
		cv2.imwrite(img_path, img_rgb)
		temp2 = img_as_ubyte(random_noise(img.astype(float), mode='gaussian'))
		img_rgb = cv2.cvtColor(temp2,cv2.COLOR_GRAY2RGB)
		cv2.imwrite(img_path2, img_rgb)


# the whole image data augmentation workflow
def workFlow(path1,path2):
	#path1: readpath--path2:savepath
	origin_cate = [path1 + '/' + x for x in os.listdir(path1) if os.path.isdir(path1 + '/' + x)]
	new_cate = [path2 + '/' + x for x in os.listdir(path1) if os.path.isdir(path1 + '/' + x)]

	for idx, folder in enumerate(origin_cate):
		nameid = 0
		img_new_path = new_cate[idx]
		isExists = os.path.exists(img_new_path)
		if not isExists:
			os.makedirs(img_new_path)
			
		files = []
		for ext in ('*.gif', '*.png', '*.jpg'):
			files.extend(glob.glob(os.path.join(folder, ext)))

		for im in files:
			print('reading the images:%s' % (im))
			temp_img = readImg(im)
			thresh2 = temp_img #imgBBoxCut(temp_img)
			t = nameid
			nameid = img_augment(thresh2,img_new_path,t)
			#nameid = single_augment_test(thresh2,img_new_path,t)
			print("over")



path1 = shf_orgImagePath
path2 = shf_trainImagebasePath
workFlow(path1,path2)