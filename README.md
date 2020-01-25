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

As observed from images it looks like bounding box sizes are varying from very small to large, so thought to use Faster RCNN network than other networks like SSD, Yolo or RetinaNet because RPN tends to give good results for smaller objects and also we get more proposals for bounding boxes. Even other networks also can get similar accuracy more inference speed. To use faster RCNN model as inference model we can go for quantization tecnique for 16FLOP or 8FLOP and also can train with architecture pruning once after reaching maximum accuracy.

One sample image with predicted bounding boxes.
![](Output.jpg)

#### Process
1. Need to configure the TensorFlow Object Detection API
2. Build an image dataset + image annotations in the TensorFlow Object Detection API Form 
3. Train a Faster R-CNN on the dataset
4. Predict the testset images.

Download or clone TFOD API - https://github.com/tensorflow/models. Complete code is in faces_count folder (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/tree/master/faces_count)

Train and test data is placed in faces_count/faces folder (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/tree/master/faces_count/faces).

Build dataset with images and annotations in TFOD API form using faces_count/build_faces_records.py (https://github.com/AbhinayReddyYarva/FaceCountingChallenge-AnalyticsVidhya/blob/master/faces_count/build_faces_records.py)



