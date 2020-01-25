In this taring folder all the pre-trained models and traineded models will be kept. 

Folder faster_rcnn_resnet101_coco_2018_01_28 contains downloaded pre-trained model and trained models will get saved in current folders

I haven't included my included my saved model here for now. Once competition ends I will add the trained. 

Below are some of the file names from current model

graph.pbtxt
model.ckpt-20000.data-00000-of-00001
model.ckpt-20000.index
model.ckpt-20000.meta

faster_rcnn_faces.config is our config and which most important in training the model. It will have how many classes, how many steps to run, fine tuned model, input data paths (traning, testing, class labels) and number of testing annotated bounding boxes.
And also can update learning rate, decay.
