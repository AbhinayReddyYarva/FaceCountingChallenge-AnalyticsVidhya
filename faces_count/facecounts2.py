from __future__ import division
import cv2
import time
import sys
import os
import csv

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1 - 5, y1), (x2 + 5, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

if __name__ == "__main__" :

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using Tensorflow ( 2.7 MB )
    DNN = "CAFFE"
    if DNN == "CAFFE":
        modelFile = os.path.join("res10_300x300_ssd_iter_140000_fp16.caffemodel")
        configFile = os.path.join("deploy.prototxt")
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = os.path.join("opencv_face_detector_uint8.pb")
        print(modelFile)
        configFile = os.path.join("opencv_face_detector.pbtxt")
        print(configFile)
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    conf_threshold = 0.7

    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    IMAGES_PATH = "faces\\image_data"

    TEST_IMAGES_CSV = "faces\\test.csv"
    TEST_IMAGES_PATH = "faces\\image_data"

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
    frame_count = 0
    tt_opencvDnn = 0
    for hasFrame, path in enumerate(testPaths):
        print(hasFrame, path)
        frame = cv2.imread(path)
        frame_count += 1

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn
        label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)
        cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("Face Detection Comparison", outOpencvDnn)

        print(len(bboxes))
        testID = path.split(os.path.sep)[-1]
        testList.append([testID, len(bboxes)])
        if frame_count == 1:
            tt_opencvDnn = 0
        k = cv2.waitKey(10)
        if k == 27:
            break
    cv2.destroyAllWindows()
    with open('submissionfacecount.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(testList)
    
