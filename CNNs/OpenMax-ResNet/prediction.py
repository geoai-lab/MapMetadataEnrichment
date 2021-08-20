from keras.models import load_model
from keras import Model
from skimage import io, transform
import os
import glob
import numpy as np
from configuration import GeoLocation_dict_continent, model_continent_path

def read_image(path, if_sample = 0, sample_num = 0):
	# 0: load the whole dataset
	# 1: random select "sample_num" samples from the whole dataset;

	imgs=[]
	label=[]
	count = 0
	img_set = glob.glob(path + '/*.jpg')
	random.shuffle(img_set)

	for im in img_set:
		count += 1
		img = io.imread(im)
		img = transform.resize(img, (224, 224, 3))

		imgs.append(img)
		label.append(os.path.basename(im))

		if if_sample == 1 and count >= sample_num:
			break

	return len(imgs),np.asarray(imgs,np.float32),label


if __name__ == '__main__':
	"""
		OpenMax works by comparing the statistical distribution of model's outputs 
		on training data and testing data. Both the training images and the test maps
		are required.
	"""

	train_image_path = '../../Training_data/image-set' # Path to the generated training images
	test_image_path = '../../Test_data/Binary_maps' # Path to test map images
	img_rows, img_cols = 224, 224  # Resolution of inputs
	channel = 3

	GeoLocation_dict_continent2 = {k: v for v, k in GeoLocation_dict_continent.items()}

	print("Begin to load ResNet model")
	model = load_model(model_continent_path)
	
	fc10 = model.get_layer("fc10").output
	new_model = Model(inputs=model.input, outputs = fc10)

	print("Begin to load test map set")
	scores_list = []

	## Process the train images and save the output vectors 
	for names in GeoLocation_dict_continent2.keys():
		category_id = int(GeoLocation_dict_continent2[names])
		imcount, data, label = read_image(train_image_path + names, 1, 1000)
		print(names)
		data = data.reshape(imcount, img_rows, img_cols, channel)

		data = data[:, :, :, ::-1]

		# Subtract ImageNet mean pixel
		data[:, :, :, 0] -= 103.939
		data[:, :, :, 1] -= 116.779
		data[:, :, :, 2] -= 123.68

		good_samples = []
		outputs = model.predict(data)
		predicts = outputs.argmax(axis=-1)

		for i_predict in range(len(predicts)):
			if predicts[i_predict] == category_id:
				good_samples.append(outputs[i_predict])

		scores_list.append(good_samples)
	scores = [np.array(x)[:, np.newaxis, :] for x in scores_list]  # (N_c, 1, C) * C
	mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)


	np.save("OpenMax/train_scores.npy", scores)
	np.save("OpenMax/mavs.npy", mavs)


	## Process the test images and save the output vectors 
	for names in GeoLocation_dict_continent2.keys():
		index = int(GeoLocation_dict_continent2[names])
		imcount, data, label = read_image(test_image_path + names)
		print(names)
		data = data.reshape(imcount, img_rows, img_cols, channel)

		data = data[:, :, :, ::-1]

		# Subtract ImageNet mean pixel
		data[:, :, :, 0] -= 103.939
		data[:, :, :, 1] -= 116.779
		data[:, :, :, 2] -= 123.68

		outputs = model.predict(data)
		scores_list.append(outputs)

	scores = [np.array(x)[:, np.newaxis, :] for x in scores_list]  # (N_c, 1, C) * C
	np.save("OpenMax/test_scores.npy", scores)
		
