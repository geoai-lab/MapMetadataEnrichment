from skimage import io, transform
import random
import tensorflow as tf
import numpy as np
import glob
import os
from configuration import GeoLocation_dict_continent, model_meta, model_weights


row = 227
col= 227
dim = 3
classes = 8

GeoLocation_dict_continent2 = {k:v for v, k in GeoLocation_dict_continent.items()}

def read_image(path):
	imgs=[]
	label=[]
	for im in glob.glob(path + '/*.jpg'):    	
		img = io.imread(im)
		img = transform.resize(img, (row, col,3))
		imgs.append(img)
		label.append(os.path.basename(im))


	return len(imgs),np.asarray(imgs,np.float32),label


with tf.Session() as sess:
	all_accuracy = 0
	for names in GeoLocation_dict_continent2.keys():
		imcount,data,label= read_image(test_image_path2+names)
		print(names)
		category_id = int(GeoLocation_dict_continent2[names])
		data=data.reshape(imcount,row,col,dim)
		saver = tf.train.import_meta_graph(model_meta)
		saver.restore(sess, model_weights)

		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name("x:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")
		logits = graph.get_tensor_by_name("logits_eval:0")

		predicts=tf.nn.softmax(logits=logits,dim=-1)

		classification_result = sess.run(logits, feed_dict={x: data, keep_prob: 1})
	
		# from matrix to list
		output = tf.argmax(classification_result, 1).eval()
		num_count=np.zeros(classes)

		for i in range(len(output)):
			print("The %s" %(label[i])+" map images classification:" + GeoLocation_dict_continent[str(output[i])])
			num_count[output[i]]=num_count[output[i]]+1

		accuracy = num_count[category_id]/len(output)
		print("The classification accuracy is:" + str(accuracy))

		print("Error analysis of model classification:")
		print("Africa: %d "%(num_count[0]))
		print("Antarctica: %d " % (num_count[1]))
		print("Asia: %d " % (num_count[2]))
		print("Europe: %d " % (num_count[3]))
		print("Global: %d " % (num_count[4]))
		print("North America: %d " % (num_count[5]))
		print("Oceania: %d " % (num_count[6]))
		print("South America: %d " % (num_count[7]))
		all_accuracy += accuracy

	print("The overall classification accuracy:")
	print(all_accuracy/classes)
