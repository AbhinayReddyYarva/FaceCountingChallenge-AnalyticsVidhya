# Face Counting Challenge - 2nd rank - Active competition - Analytics Vidhya Hackathon

Link - https://datahack.analyticsvidhya.com/contest/vista-codefest-computer-vision-1/

#### Problem Statement
People detection and head counting is one of the classical albeit challenging computer vision application. For this problem, given a group selfie/photo, you are required to count the number of heads present in the picture. You are provided with a training set of images with coordinates of bounding box and head count for each image and need to predict the headcount for each image in the test set.

#### Evaluation Metric
The evaluation metric for this competition is RMSE (root mean squared error) over the head counts predicted for test images.

#### Train and test data
Under Problem Statement section - https://datahack.analyticsvidhya.com/contest/vista-codefest-computer-vision-1/ 

#### Hardware
Cloud instance with GPU (GCP)
GPU -> 4, P100 16GB GPUs
RAM -> 120GB (64GB RAM is minimum to have 4 P100 16GB GPUs in GCP instance). 

#### Framework 
Tensorflow Object Detection API - https://github.com/tensorflow/models

### Journey
Initially I tried with mostly used and popular networks used in OpenCV to make it as base RMSE score.
1. FP16 version of the original caffe implementation (res10_300x300_ssd_iter_140000_fp16.caffemodel)
2. 8 bit Quantized version using Tensorflow (opencv_face_detector_uint8.pb)

Using res10_300x300_ssd_iter_140000_fp16.caffemodel got 2.76 RMSE score
![](faces_count/AV-Scores/res10ssd.JPG)

Using opencv_face_detector_uint8.pb got 2.73 RMSE score
![](faces_count/AV-Scores/opencvfacedetector.JPG)

Implimentation of above models can be found in faces_count/facecounts2.py (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/blob/master/faces_count/facecounts2.py)

To view the images with bounding boxes of training set images please go through the file faces_count/visual.py (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/blob/master/faces_count/visual.py) 
When observed the bounding boxes of training dataset there are many faces with blur, side, straight half, side half, small (captured from far). Below are some sameple images with bounding boxes. 
![](BBoxImages/BlurFace.JPG)
![](BBoxImages/SideFace.JPG)
![](BBoxImages/HalfFace.JPG)
![](BBoxImages/SideHalf.JPG)
![](BBoxImages/SmallBlur.JPG)

Model res10_300x300_ssd_iter_140000_fp16.caffemodel and opencv_face_detector_uint8.pb will work well if face is near to camera but will not work well on half face or blur or small.

As observed from images it looks like bounding box sizes are varying from very small to large, so thought to use Faster RCNN network than other networks like SSD, Yolo or RetinaNet because RPN tends to give good results for localizing smaller objects as we get more proposals for bounding boxes. Even other networks also can get similar accuracy more inference speed. To use faster RCNN model as inference model we can go for quantization tecnique for 16FLOP or 8FLOP and also can train with architecture pruning once after reaching maximum accuracy.

One sample image with predicted bounding boxes.
![](Output.jpg)

#### Traning Process
Complete trainig took around 4 hours using 4 P100 16GB GPUs and 120GB RAM in Cloud instance (GCP) for 20000 steps usually it is suggested to run for 50000 - 100000 steps.

1. Configure the TensorFlow Object Detection API (To use 
2. Build an image dataset + image annotations in the TensorFlow Object Detection API Form 
3. Train a Faster R-CNN on the dataset
4. Predict the testset images.

Download or clone TFOD API - https://github.com/tensorflow/models. Complete code is in faces_count folder (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/tree/master/faces_count) and this folder is placed in models/research folder, so that it would be easy to use code from object_detection folder (models/research/object_detection).
##### models/research/faces_counts)

Train and test data is placed in faces_count/faces folder (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/tree/master/faces_count/faces).

Build dataset with images and annotations in TFOD API form using faces_count/build_faces_records.py (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/blob/master/faces_count/build_faces_records.py)

Before running above script need to add TFOD API's models/research folder and models/research/slim folder to $PYTHONPATH

I have used faster rcnn model with resnet101 architecture trained on coco dataset. Downloaded the pre-trained model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and places it in faces_count/faces/experiments/training folder (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/tree/master/faces_count/faces/experiments/training) 

Config file for TFOD API is created - faces_count/faces/experiments/training/faster_rcnn_faces.config (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/blob/master/faces_count/faces/experiments/training/faster_rcnn_faces.config). Sample config files can be downloaded from https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
More info - faces_count/faces/experiments/training/README.md (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/blob/master/faces_count/faces/experiments/training/README.md)

Got 95.5 mAP score for 0.5 IoU after training. 

Model is working good when verified on test image to visualize the result. 
![](Output.jpg)

When tested on test data got 0.918 RMSE score which is much better than base bench mark scores. 
![](faces_count/AV-Scores/FasterRCNNRMSE.JPG)

#### Commands
time python3 faces_count/build_faces_records.py

python3 object_detection/model_main.py --pipeline_config_path=faces_count/faces/experiments/training/faster_rcnn_faces.config --model_dir=faces_count/faces/experiments/training --num_train_steps=20000 --sample_1_of_n_eval_examples=1 --alsologtostderr

python3 object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faces_count/faces/experiments/training/faster_rcnn_faces.config --trained_checkpoint_prefix=faces_count/faces/experiments/training/model.ckpt-20000 --output_directory=faces_count/faces/experiments/exported_model

python3 faces_count/predict.py --model faces_count/faces/experiments/exported_model/frozen_inference_graph.pb --labels faces_count/faces/records/classes.pbtxt --image faces_count/faces/test.csv --num-classes 1

![](faces_count/AV-Scores/FasterRCNN.JPG)
