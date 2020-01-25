
# import the necessary packages
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
import os
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="base path for frozen checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True,
	help="labels file")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-n", "--num-classes", type=int, required=True,
	help="# of class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# initialize a set of colors for our class labels
COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))

# initialize the model
model = tf.Graph()

# create a context manager that makes this model the default one for
# execution
with model.as_default():
	# initialize the graph definition
	graphDef = tf.GraphDef()
	print("after graph")
	# load the graph from disk
	with tf.gfile.GFile(args["model"], "rb") as f:
		serializedGraph = f.read()
		graphDef.ParseFromString(serializedGraph)
		tf.import_graph_def(graphDef, name="")
print("after gfile")
# load the class labels from disk
labelMap = label_map_util.load_labelmap(args["labels"])
categories = label_map_util.convert_label_map_to_categories(
	labelMap, max_num_classes=args["num_classes"],
	use_display_name=True)
categoryIdx = label_map_util.create_category_index(categories)
print(categoryIdx)
# create a session to perform inference
with model.as_default():
	with tf.Session(graph=model) as sess:
		TEST_IMAGES_CSV = args["image"]
		TEST_IMAGES_PATH = "faces_count/faces/image_data"
		print(TEST_IMAGES_CSV)
		testPaths = [] #list(paths.list_images(TEST_IMAGES_PATH))
		t = open(TEST_IMAGES_CSV)
		t.__next__() # f.next() for Python 2.7
		tlist = list(t) 
		for i, row in enumerate(tlist):
			# extract the image and label from the row
			image = row.rstrip()
			imagepath = os.path.join(TEST_IMAGES_PATH, image)  
			testPaths.append(imagepath)
		t.close()

		print(len(testPaths))
		testList = [["Name","HeadCount"]]
		for path in testPaths:
			# tensor
			imageTensor = model.get_tensor_by_name("image_tensor:0")
			boxesTensor = model.get_tensor_by_name("detection_boxes:0")

			# for each bounding box we would like to know the score
			# (i.e., probability) and class label
			scoresTensor = model.get_tensor_by_name("detection_scores:0")
			classesTensor = model.get_tensor_by_name("detection_classes:0")
			numDetections = model.get_tensor_by_name("num_detections:0")

			# load the image from disk
			image = cv2.imread(path)
			(H, W) = image.shape[:2]

			# check to see if we should resize along the width
			if W > H and W > 1000:
				image = imutils.resize(image, width=1000)

			# otherwise, check to see if we should resize along the
			# height
			elif H > W and H > 1000:
				image = imutils.resize(image, height=1000)

			# prepare the image for detection
			(H, W) = image.shape[:2]
			output = image.copy()
			image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
			image = np.expand_dims(image, axis=0)

			# perform inference and compute the bounding boxes,
			# probabilities, and class labels
			(boxes, scores, labels, N) = sess.run(
				[boxesTensor, scoresTensor, classesTensor, numDetections],
				feed_dict={imageTensor: image})

			# squeeze the lists into a single dimension
			boxes = np.squeeze(boxes)
			scores = np.squeeze(scores)
			labels = np.squeeze(labels)

			count = 0 # loop over the bounding box predictions
			for (box, score, label) in zip(boxes, scores, labels):
				# if the predicted probability is less than the minimum
				# confidence, ignore it
				if score < args["min_confidence"]:
					continue

				# counting number of boxes
				count = count + 1

			# draw the prediction on the output image
			testID = path.split(os.path.sep)[-1]
			print([testID, count])
			testList.append([testID, count])
		#print(testList)
		with open('submissionfacecount.csv', 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerows(testList)