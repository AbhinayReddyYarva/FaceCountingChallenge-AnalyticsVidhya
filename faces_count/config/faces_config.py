# import the necessary packages
import os

# initialize the base path for the LISA dataset
BASE_PATH = "faces_count/faces"
IMAGES_PATH = "faces_count/faces/image_data"

# build the path to the annotations file
ANNOT_PATH = os.path.sep.join([BASE_PATH, "bbox_train.csv"])
TRAIN_CSV = os.path.sep.join([BASE_PATH, "train.csv"])
TEST_CSV = os.path.sep.join([BASE_PATH, "test.csv"])

# build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = os.path.sep.join([BASE_PATH,
	"records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH,
	"records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH,
	"records/classes.pbtxt"])

# initialize the test split size
TEST_SIZE = 0.25

# initialize the class labels dictionary
CLASSES = {"face": 1}