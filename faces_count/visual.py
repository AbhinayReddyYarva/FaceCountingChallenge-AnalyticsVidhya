# USAGE
# python annotate.py --input Images/directory --csv csv/file

# import the necessary packages
from imutils import paths
import argparse
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-a", "--csv", required=True,
	help="path to csv file of images names and annotations")
args = vars(ap.parse_args())

# grab the image paths
imagePaths = list(paths.list_images(args["input"]))
print("[INFO] loading train csv file to create image paths and labels data...")
f = open(args["csv"])
f.__next__() # f.next() for Python 2.7
imagePaths = []
widths = []
heights = []
xmins = []
ymins = []
xmaxs = []
ymaxs =[]
flist = list(f) 
for i, row in enumerate(flist):
    # extract the image and label from the row
    (name, width, height, xmin, ymin, xmax, ymax) = row.strip().split(",")
    imagepath = os.path.join(args["input"], name)
    imagePaths.append(imagepath)
    widths.append(width)
    heights.append(height)
    xmins.append(xmin)
    ymins.append(ymin)
    xmaxs.append(xmax)
    ymaxs.append(ymax)
f.close()

print(len(imagePaths))

for name, width, height, xmin, ymin, xmax, ymax in zip(imagePaths, widths, heights, xmins, ymins, xmaxs, ymaxs):
    image = cv2.imread(name)
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow(name, image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key & 0xff == ord('q'):
        break
    


